"""Estimator-style public API and backend registry."""

from __future__ import annotations

import importlib
import math
from pathlib import Path
import time
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np

from .problem import (
    MarkowitzInstance,
    evaluate_solution,
    load_instance_bundle,
)
from .result import RelaxationResult
from .native import (
    AUTO_IMPLEMENTATION,
    FIRST_ORDER_BACKENDS,
    _normalize_implementation,
    load_first_order_module,
)
from .state import RelaxationState


DEFAULT_BACKEND = "fista"
DEFAULT_PAVA = "partial_sort"
DEFAULT_PDHG_VARIANT = "metric-linesearch-restart"
DEFAULT_FISTA_RESTART = "gradient"
DEFAULT_FISTA_PROX_ORACLE = "auto"
DEFAULT_IMPLEMENTATION = "auto"

BACKENDS = (
    "fista",
    "gurobi",
    "mosek",
    "jump",
    "gurobi.python",
    "gurobi.julia",
    "mosek.python",
    "mosek.julia",
    "jump.julia",
)
# Kept callable for backward compatibility with research scripts, but no
# longer advertised as part of the student-facing package surface.
_COMPATIBILITY_BACKENDS = ("pdhg", "scsdg")
_BACKEND_ALIASES = {
    "gurobi": "gurobi.python",
    "mosek": "mosek.python",
    "jump": "jump.julia",
}
ProblemLike = Union[MarkowitzInstance, str, Path]


def registered_backends() -> tuple[str, ...]:
    """Return registered backend identifiers without probing licenses."""
    return BACKENDS


def available_backends() -> tuple[str, ...]:
    """Compatibility alias for :func:`registered_backends`.

    A registered optional backend can still be unavailable at solve time
    because its package, runtime, or license is missing.
    """
    return registered_backends()


def _problem(value: ProblemLike) -> MarkowitzInstance:
    if isinstance(value, MarkowitzInstance):
        value.validate()
        return value
    return load_instance_bundle(value)


def _backend_solver(
    backend: str,
    implementation: str = "python",
) -> Any:
    resolved = _BACKEND_ALIASES.get(backend, backend)
    if resolved in FIRST_ORDER_BACKENDS:
        module, _ = load_first_order_module(resolved, implementation)
        return getattr(module, f"solve_{resolved}")
    family, language = resolved.split(".", 1)
    return importlib.import_module(
        f"sksfolio.relaxation.{family}.{language}"
    ).solve


def _normalize_solver_objective_bound(
    raw: Dict[str, Any],
    solver: str,
) -> None:
    """Expose a commercial solver's numerical bound separately."""
    details = raw.get("solver_details", {})
    value = raw.get(
        "solver_objective_bound",
        details.get("objective_bound")
        if isinstance(details, Mapping)
        else None,
    )
    try:
        bound = float(value)
    except (TypeError, ValueError):
        return
    if not math.isfinite(bound):
        return
    raw["solver_objective_bound"] = bound
    raw.setdefault(
        "solver_objective_bound_source",
        {
            "gurobi": "gurobi_obj_bound",
            "mosek": "mosek_dual_objective",
        }.get(solver, "solver_reported_objective_bound"),
    )
    raw.setdefault("solver_objective_bound_units", "original_objective")
    raw.setdefault("solver_objective_bound_numerically_reported", True)
    raw.setdefault(
        "solver_objective_bound_floating_point_certified",
        False,
    )
    raw.setdefault(
        "solver_objective_bound_note",
        "solver-reported floating-point bound; not the saved "
        "recomputable Fenchel certificate",
    )


def _attach_safe_dual_certificate(
    instance: MarkowitzInstance,
    weights: np.ndarray,
    raw: Dict[str, Any],
) -> None:
    """Add a recomputable Fenchel bound when a backend omitted one."""
    if (
        raw.get("best_dual_lower_bound") is not None
        and raw.get("dual_bound_factor") is not None
        and raw.get("dual_bound_constraint_original") is not None
    ):
        return
    from .pdhg.safe_dual import SafeDualEvaluator

    factor = np.asarray(instance.B.T @ weights, dtype=float).reshape(-1)
    constraint = np.zeros(instance.rows, dtype=float)
    certificate = SafeDualEvaluator(instance).evaluate(
        factor,
        constraint,
    )
    bound = certificate["dual_bound"]
    if bound is None:
        return
    raw.update(
        {
            "dual_bound": bound,
            "best_dual_lower_bound": bound,
            "dual_bound_available": True,
            "dual_bound_safe_in_exact_arithmetic": certificate[
                "safe_in_exact_arithmetic"
            ],
            "dual_bound_floating_point_certified": certificate[
                "floating_point_certified"
            ],
            "dual_bound_factor": certificate["factor_dual"],
            "dual_bound_constraint_scaled": certificate[
                "constraint_dual_scaled"
            ],
            "dual_bound_constraint_original": certificate[
                "constraint_dual_original"
            ],
            "maximum_dual_domain_correction": certificate[
                "dual_domain_correction"
            ],
            "dual_bound_kind": "fenchel_weak_duality_lower_bound",
            "dual_bound_source": (
                "sksfolio_postprocessed_zero_constraint_multiplier"
            ),
            "dual_bound_units": "original_objective",
            "dual_bound_formula": (
                "-0.5*||p||^2 - support_[lower,upper](q) "
                "- (perspective_weight*G_k)^*("
                "return_reward*mu - B*p - C.T*q)"
            ),
        }
    )


def solve_relaxation(
    problem: ProblemLike,
    backend: str = DEFAULT_BACKEND,
    *,
    variant: str = DEFAULT_PDHG_VARIANT,
    pava: str = DEFAULT_PAVA,
    implementation: str = DEFAULT_IMPLEMENTATION,
    warm_start: Optional[Any] = None,
    options: Optional[Mapping[str, Any]] = None,
) -> RelaxationResult:
    """Solve one perspective relaxation with an interchangeable backend.

    The benchmark-selected default is line-search FISTA with automatic
    L-BFGS proximal-oracle selection, native partial-sort PAVA, and gradient
    restart. Whole-module Cython showed no robust warm-time improvement over
    this already-native numerical path, so ``implementation="auto"``
    currently selects Python orchestration. Set
    ``implementation="python"`` for the source implementation or
    ``implementation="native"`` to require the compiled implementation.
    ``warm_start`` accepts a prior result or :class:`RelaxationState` and is
    safe to reuse after branch rows change because multipliers are matched by
    constraint name and acceleration is restarted.
    The short ``gurobi`` and ``mosek`` names select their native Python
    APIs; append ``.julia`` to request their JuMP wrappers. The ``jump``
    backend accepts an optimizer selector in ``options`` and defaults to
    Clarabel.
    """
    if backend not in BACKENDS + _COMPATIBILITY_BACKENDS:
        raise ValueError(
            "backend must be one of " + ", ".join(BACKENDS)
        )
    instance = _problem(problem)
    supplied: Dict[str, Any] = dict(options or {})
    resolved = _BACKEND_ALIASES.get(backend, backend)
    if resolved in FIRST_ORDER_BACKENDS:
        requested_implementation = _normalize_implementation(implementation)
        actual_implementation = requested_implementation
        if requested_implementation == "auto":
            actual_implementation = AUTO_IMPLEMENTATION[resolved]
        solver = _backend_solver(resolved, actual_implementation)
    else:
        if str(implementation).lower() != "auto":
            raise ValueError(
                "implementation= applies only to first-order backends; "
                "select commercial wrapper languages with the backend name"
            )
        actual_implementation = "julia" if resolved.endswith(".julia") else "python"
        solver = _backend_solver(backend)
    if warm_start is False:
        if resolved not in FIRST_ORDER_BACKENDS:
            supplied.setdefault("warm_start", False)
    elif warm_start is not None:
        if "warm_start" in supplied or "initial_state" in supplied:
            raise ValueError(
                "pass warm_start either as a named argument or in options, "
                "not both"
            )
        state = RelaxationState.coerce(warm_start)
        if resolved.endswith(".julia"):
            supplied["warm_start"] = True
            supplied["initial_x"] = state.compatible_primal(
                instance
            ).tolist()
        else:
            supplied["warm_start"] = state
    start = time.perf_counter()
    if backend in {"pdhg", "fista", "scsdg"}:
        if "pava_backend" in supplied:
            raise ValueError(
                "pass the PAVA choice with pava=, not "
                "options['pava_backend']"
            )
        supplied["pava_backend"] = pava
        if backend == "pdhg":
            raw = dict(
                solver(
                    instance,
                    variant=variant,
                    options=supplied,
                )
            )
        elif backend == "fista":
            supplied.setdefault(
                "prox_oracle",
                DEFAULT_FISTA_PROX_ORACLE,
            )
            if (
                "restart_strategy" not in supplied
                and "adaptive_restart" not in supplied
            ):
                supplied["restart_strategy"] = DEFAULT_FISTA_RESTART
            raw = dict(solver(instance, options=supplied))
        else:
            raw = dict(solver(instance, options=supplied))
    else:
        raw = dict(solver(instance, supplied))
    elapsed = time.perf_counter() - start
    postprocess_start = time.perf_counter()

    solver_details = raw.get("solver_details", {})
    if "warm_start_used" not in raw:
        if (
            isinstance(solver_details, Mapping)
            and "warm_start_used" in solver_details
        ):
            raw["warm_start_used"] = bool(
                solver_details["warm_start_used"]
            )
        elif "warm_start_applied" in raw:
            raw["warm_start_used"] = bool(raw["warm_start_applied"])

    family, _, language = resolved.partition(".")
    _normalize_solver_objective_bound(raw, family)
    raw["backend"] = backend
    raw.setdefault("canonical_backend", resolved)
    raw.setdefault(
        "language",
        (
            "cython"
            if actual_implementation == "native"
            else language or "python"
        ),
    )
    raw.setdefault("implementation", actual_implementation)
    raw.setdefault(
        "implementation_selection",
        (
            "matched_benchmark_auto_policy"
            if str(implementation).lower() == "auto"
            else "explicit"
        ),
    )
    raw.setdefault(
        "native_solver_core",
        actual_implementation == "native",
    )
    raw.setdefault(
        "compiled_solver_module",
        actual_implementation == "native",
    )
    raw.setdefault("native_hot_loop_typed", False)
    raw.setdefault("gil_free_solver_loop", False)
    raw.setdefault("solver", family)
    raw.setdefault("backend_call_seconds", elapsed)
    if raw.get("x") is not None:
        weights = np.asarray(raw["x"], dtype=float).reshape(-1)
        if weights.shape != (instance.dimension,):
            raise ValueError(
                f"{backend} returned weights with shape {weights.shape}"
            )
        try:
            diagnostic_tolerance = max(
                1e-8,
                float(raw.get("effective_tolerance", 1e-8)),
            )
        except (TypeError, ValueError):
            diagnostic_tolerance = 1e-8
        raw["x"] = weights
        raw["has_solution"] = True
        raw["diagnostic_domain_tolerance"] = diagnostic_tolerance
        raw["diagnostics"] = evaluate_solution(
            instance,
            weights,
            domain_tolerance=diagnostic_tolerance,
        )
        raw["external_objective"] = raw["diagnostics"]["objective"]
        _attach_safe_dual_certificate(instance, weights, raw)
        maximum_violation = float(
            raw["diagnostics"]["violations"]["maximum"]
        )
        objective = raw["external_objective"]
        primal_feasible = bool(
            objective is not None
            and math.isfinite(float(objective))
            and maximum_violation <= diagnostic_tolerance
        )
        raw["primal_feasible"] = primal_feasible
        raw["primal_upper_bound"] = (
            float(objective) if primal_feasible else None
        )

        restart_value = raw.get("restart_state", raw.get("state"))
        if restart_value is None:
            restart_factor = raw.get("dual_bound_factor")
            if restart_factor is None:
                restart_factor = instance.B.T @ weights
            restart_constraint = raw.get(
                "dual_bound_constraint_original"
            )
            if restart_constraint is None:
                restart_constraint = np.zeros(instance.rows)
            raw["restart_state"] = RelaxationState(
                backend=family,
                implementation=actual_implementation,
                dimension=instance.dimension,
                k=instance.k,
                constraint_ids=tuple(instance.constraint_names),
                x=weights,
                factor_dual=np.asarray(
                    restart_factor,
                    dtype=float,
                ).reshape(-1),
                constraint_dual=np.asarray(
                    restart_constraint,
                    dtype=float,
                ).reshape(-1),
            ).to_dict(copy=False)
        else:
            restart_state = RelaxationState.coerce(restart_value)
            if restart_state is None:
                raise RuntimeError("backend returned an empty restart state")
            if (
                restart_state.factor_dual is None
                or restart_state.constraint_dual is None
            ):
                payload = restart_state.to_dict(copy=False)
                if restart_state.factor_dual is None:
                    restart_factor = raw.get("dual_bound_factor")
                    if restart_factor is None:
                        restart_factor = instance.B.T @ weights
                    payload["factor_dual"] = np.asarray(
                        restart_factor,
                        dtype=float,
                    ).reshape(-1)
                if restart_state.constraint_dual is None:
                    restart_constraint = raw.get(
                        "dual_bound_constraint_original"
                    )
                    if restart_constraint is None:
                        restart_constraint = np.zeros(instance.rows)
                    payload["constraint_dual"] = np.asarray(
                        restart_constraint,
                        dtype=float,
                    ).reshape(-1)
                raw["restart_state"] = RelaxationState.from_mapping(
                    payload
                ).to_dict(copy=False)
        bound = raw.get("best_dual_lower_bound")
        raw["bound_consistency_violation"] = (
            max(
                float(bound) - float(raw["primal_upper_bound"]),
                0.0,
            )
            if bound is not None
            and raw["primal_upper_bound"] is not None
            else None
        )
        raw["primal_safe_gap"] = (
            float(raw["primal_upper_bound"]) - float(bound)
            if bound is not None
            and raw["primal_upper_bound"] is not None
            else None
        )
    else:
        raw.setdefault("has_solution", False)
        raw.setdefault("diagnostics", {})
        raw.setdefault("primal_feasible", False)
        raw.setdefault("primal_upper_bound", None)
        raw.setdefault("primal_safe_gap", None)
    raw["api_postprocess_seconds"] = (
        time.perf_counter() - postprocess_start
    )
    raw["end_to_end_seconds"] = time.perf_counter() - start
    raw["wrapper_seconds"] = raw["end_to_end_seconds"]
    return RelaxationResult(raw)


class PerspectiveRelaxation:
    """Small estimator-style wrapper inspired by scikit-learn/skfolio."""

    def __init__(
        self,
        backend: str = DEFAULT_BACKEND,
        *,
        variant: str = DEFAULT_PDHG_VARIANT,
        pava: str = DEFAULT_PAVA,
        implementation: str = DEFAULT_IMPLEMENTATION,
        warm_start: Optional[Any] = None,
        solver_params: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.backend = backend
        self.variant = variant
        self.pava = pava
        self.implementation = implementation
        self.warm_start = warm_start
        self.solver_params = dict(solver_params or {})

    def fit(
        self,
        problem: ProblemLike,
        *,
        warm_start: Optional[Any] = None,
    ) -> "PerspectiveRelaxation":
        self.problem_ = _problem(problem)
        selected_warm_start = (
            self.warm_start if warm_start is None else warm_start
        )
        if selected_warm_start is True:
            selected_warm_start = getattr(self, "state_", None)
        self.result_ = solve_relaxation(
            self.problem_,
            self.backend,
            variant=self.variant,
            pava=self.pava,
            implementation=self.implementation,
            warm_start=selected_warm_start,
            options=self.solver_params,
        )
        if self.result_.weights is None:
            raise RuntimeError(
                f"{self.backend} returned no portfolio: "
                f"{self.result_.status}"
            )
        self.weights_ = self.result_.weights.copy()
        self.state_ = self.result_.state
        return self


__all__ = [
    "BACKENDS",
    "DEFAULT_BACKEND",
    "DEFAULT_FISTA_PROX_ORACLE",
    "DEFAULT_FISTA_RESTART",
    "DEFAULT_IMPLEMENTATION",
    "DEFAULT_PAVA",
    "DEFAULT_PDHG_VARIANT",
    "PerspectiveRelaxation",
    "available_backends",
    "registered_backends",
    "solve_relaxation",
]
