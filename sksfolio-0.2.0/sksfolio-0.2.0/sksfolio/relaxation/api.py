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


DEFAULT_BACKEND = "fista"
DEFAULT_PAVA = "partial_sort"
DEFAULT_PDHG_VARIANT = "metric-linesearch-restart"
DEFAULT_FISTA_RESTART = "gradient"
DEFAULT_FISTA_PROX_ORACLE = "auto"

BACKENDS = (
    "fista",
    "pdhg",
    "gurobi",
    "mosek",
    "gurobi.python",
    "gurobi.julia",
    "mosek.python",
    "mosek.julia",
)
_BACKEND_ALIASES = {
    "gurobi": "gurobi.python",
    "mosek": "mosek.python",
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


def _backend_solver(backend: str) -> Any:
    resolved = _BACKEND_ALIASES.get(backend, backend)
    if resolved == "pdhg":
        return importlib.import_module(
            "sksfolio.relaxation.pdhg.solver"
        ).solve_pdhg
    if resolved == "fista":
        return importlib.import_module(
            "sksfolio.relaxation.fista.solver"
        ).solve_fista
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
    options: Optional[Mapping[str, Any]] = None,
) -> RelaxationResult:
    """Solve one perspective relaxation with an interchangeable backend.

    The benchmark-selected default is line-search FISTA with automatic
    proximal-oracle selection, partial-sort PAVA, and gradient restart.
    The short ``gurobi`` and ``mosek`` names select their native Python
    APIs; append ``.julia`` to request the JuMP wrappers.
    """
    if backend not in BACKENDS:
        raise ValueError(
            "backend must be one of " + ", ".join(BACKENDS)
        )
    instance = _problem(problem)
    supplied: Dict[str, Any] = dict(options or {})
    solver = _backend_solver(backend)
    start = time.perf_counter()
    if backend in {"pdhg", "fista"}:
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
        else:
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
        raw = dict(solver(instance, supplied))
    elapsed = time.perf_counter() - start
    postprocess_start = time.perf_counter()

    resolved = _BACKEND_ALIASES.get(backend, backend)
    family, _, language = resolved.partition(".")
    _normalize_solver_objective_bound(raw, family)
    raw["backend"] = backend
    raw.setdefault("canonical_backend", resolved)
    raw.setdefault(
        "language",
        language or "python",
    )
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
        objective = raw["external_objective"]
        bound = raw.get("best_dual_lower_bound")
        raw.setdefault(
            "bound_consistency_violation",
            (
                max(float(bound) - float(objective), 0.0)
                if bound is not None and objective is not None
                else None
            ),
        )
    else:
        raw.setdefault("has_solution", False)
        raw.setdefault("diagnostics", {})
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
        solver_params: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.backend = backend
        self.variant = variant
        self.pava = pava
        self.solver_params = dict(solver_params or {})

    def fit(self, problem: ProblemLike) -> "PerspectiveRelaxation":
        self.problem_ = _problem(problem)
        self.result_ = solve_relaxation(
            self.problem_,
            self.backend,
            variant=self.variant,
            pava=self.pava,
            options=self.solver_params,
        )
        if self.result_.weights is None:
            raise RuntimeError(
                f"{self.backend} returned no portfolio: "
                f"{self.result_.status}"
            )
        self.weights_ = self.result_.weights.copy()
        return self


__all__ = [
    "BACKENDS",
    "DEFAULT_BACKEND",
    "DEFAULT_FISTA_PROX_ORACLE",
    "DEFAULT_FISTA_RESTART",
    "DEFAULT_PAVA",
    "DEFAULT_PDHG_VARIANT",
    "PerspectiveRelaxation",
    "available_backends",
    "registered_backends",
    "solve_relaxation",
]
