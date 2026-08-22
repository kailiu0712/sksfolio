"""Public API for the two corrected perspective-relaxation algorithms."""

from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np

from .fista.solver import solve_fista
from .problem import MarkowitzInstance, evaluate_solution, load_instance_bundle
from .result import RelaxationResult
from .safe_dual import SafeDualEvaluator
from .state import RelaxationState


CORRECTED_ALGORITHMS = ("corrected_fista", "corrected_lbfgs")
BACKENDS = CORRECTED_ALGORITHMS
DEFAULT_BACKEND = "corrected_lbfgs"
DEFAULT_PAVA = "partial_sort"
DEFAULT_FISTA_RESTART = "gradient"
ProblemLike = Union[MarkowitzInstance, str, Path]

_PROX_ORACLE = {
    "corrected_fista": "dual_fista",
    "corrected_lbfgs": "dual_lbfgs",
}


def registered_backends() -> tuple[str, ...]:
    """Return the only two supported continuous-relaxation algorithms."""
    return BACKENDS


def available_backends() -> tuple[str, ...]:
    """Return algorithms available in the base installation."""
    return BACKENDS


def _problem(value: ProblemLike) -> MarkowitzInstance:
    if isinstance(value, MarkowitzInstance):
        value.validate()
        return value
    return load_instance_bundle(value)


def _attach_safe_dual_certificate(
    instance: MarkowitzInstance,
    weights: np.ndarray,
    raw: Dict[str, Any],
) -> None:
    """Add a recomputable certificate if a solver stopped before saving one."""
    if (
        raw.get("best_dual_lower_bound") is not None
        and raw.get("dual_bound_factor") is not None
        and raw.get("dual_bound_constraint_original") is not None
    ):
        return
    factor = np.asarray(instance.B.T @ weights, dtype=float).reshape(-1)
    certificate = SafeDualEvaluator(instance).evaluate(
        factor,
        np.zeros(instance.rows),
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
            "dual_bound_floating_point_certified": False,
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
            "dual_bound_source": "sksfolio_postprocessed_zero_row_multiplier",
            "dual_bound_units": "original_objective",
        }
    )


def solve_relaxation(
    problem: ProblemLike,
    backend: str = DEFAULT_BACKEND,
    *,
    pava: str = DEFAULT_PAVA,
    warm_start: Optional[Any] = None,
    options: Optional[Mapping[str, Any]] = None,
) -> RelaxationResult:
    """Solve with corrected dual-FISTA or corrected dual-L-BFGS.

    ``backend`` is either ``"corrected_fista"`` or
    ``"corrected_lbfgs"``. The latter independently checks the L-BFGS-B
    proximal residual and falls back to corrected dual-FISTA when needed.
    """
    algorithm = str(backend).strip().lower().replace("-", "_")
    if algorithm not in BACKENDS:
        raise ValueError("backend must be one of " + ", ".join(BACKENDS))
    instance = _problem(problem)
    supplied: Dict[str, Any] = dict(options or {})
    controlled = {
        "algorithm_variant",
        "prox_oracle",
        "pava_backend",
        "implementation",
    }
    conflicts = sorted(controlled.intersection(supplied))
    if conflicts:
        raise ValueError(
            "algorithm selection is controlled by backend; remove options: "
            + ", ".join(conflicts)
        )
    if warm_start is not None and warm_start is not False:
        if "warm_start" in supplied or "initial_state" in supplied:
            raise ValueError(
                "pass warm_start either directly or in options, not both"
            )
        supplied["warm_start"] = RelaxationState.coerce(warm_start)
    supplied["pava_backend"] = pava
    supplied["prox_oracle"] = _PROX_ORACLE[algorithm]
    if (
        "restart_strategy" not in supplied
        and "adaptive_restart" not in supplied
    ):
        supplied["restart_strategy"] = DEFAULT_FISTA_RESTART

    start = time.perf_counter()
    raw = dict(solve_fista(instance, options=supplied))
    backend_seconds = time.perf_counter() - start
    postprocess_start = time.perf_counter()

    raw["backend"] = algorithm
    raw["canonical_backend"] = algorithm
    raw["algorithm"] = algorithm
    raw["solver"] = algorithm
    raw["language"] = "python"
    raw["implementation"] = "python"
    raw["native_solver_core"] = False
    raw.setdefault("backend_call_seconds", backend_seconds)

    if raw.get("x") is not None:
        weights = np.asarray(raw["x"], dtype=float).reshape(-1)
        if weights.shape != (instance.dimension,):
            raise ValueError(
                f"{algorithm} returned weights with shape {weights.shape}"
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
            restart_constraint = raw.get("dual_bound_constraint_original")
            if restart_constraint is None:
                restart_constraint = np.zeros(instance.rows)
            state = RelaxationState(
                backend=algorithm,
                implementation="python",
                dimension=instance.dimension,
                k=instance.k,
                constraint_ids=tuple(instance.constraint_names),
                x=weights,
                factor_dual=restart_factor,
                constraint_dual=restart_constraint,
            )
        else:
            state = RelaxationState.coerce(restart_value)
            if state is None:
                raise RuntimeError("solver returned an empty restart state")
            state.backend = algorithm
            state.implementation = "python"
        raw["restart_state"] = state.to_dict(copy=False)

        bound = raw.get("best_dual_lower_bound")
        upper = raw["primal_upper_bound"]
        raw["bound_consistency_violation"] = (
            max(float(bound) - float(upper), 0.0)
            if bound is not None and upper is not None
            else None
        )
        raw["primal_safe_gap"] = (
            float(upper) - float(bound)
            if bound is not None and upper is not None
            else None
        )
    else:
        raw.setdefault("has_solution", False)
        raw.setdefault("diagnostics", {})
        raw.setdefault("primal_feasible", False)
        raw.setdefault("primal_upper_bound", None)
        raw.setdefault("primal_safe_gap", None)

    raw["api_postprocess_seconds"] = time.perf_counter() - postprocess_start
    raw["end_to_end_seconds"] = time.perf_counter() - start
    raw["wrapper_seconds"] = raw["end_to_end_seconds"]
    return RelaxationResult(raw)


class PerspectiveRelaxation:
    """Estimator-style wrapper for one corrected relaxation algorithm."""

    def __init__(
        self,
        backend: str = DEFAULT_BACKEND,
        *,
        pava: str = DEFAULT_PAVA,
        warm_start: Optional[Any] = None,
        solver_params: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.backend = backend
        self.pava = pava
        self.warm_start = warm_start
        self.solver_params = dict(solver_params or {})

    def fit(
        self,
        problem: ProblemLike,
        *,
        warm_start: Optional[Any] = None,
    ) -> "PerspectiveRelaxation":
        self.problem_ = _problem(problem)
        selected = self.warm_start if warm_start is None else warm_start
        if selected is True:
            selected = getattr(self, "state_", None)
        self.result_ = solve_relaxation(
            self.problem_,
            self.backend,
            pava=self.pava,
            warm_start=selected,
            options=self.solver_params,
        )
        if self.result_.weights is None:
            raise RuntimeError(
                f"{self.backend} returned no portfolio: {self.result_.status}"
            )
        self.weights_ = self.result_.weights.copy()
        self.state_ = self.result_.state
        return self


__all__ = [
    "BACKENDS",
    "CORRECTED_ALGORITHMS",
    "DEFAULT_BACKEND",
    "DEFAULT_FISTA_RESTART",
    "DEFAULT_PAVA",
    "PerspectiveRelaxation",
    "available_backends",
    "registered_backends",
    "solve_relaxation",
]
