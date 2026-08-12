"""Fixed-support convex QP solves used by all incumbent heuristics."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import math
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from scipy import optimize, sparse

from ..relaxation.problem import MarkowitzInstance
from .evaluation import evaluate_incumbent, sparse_objective


RESTRICTED_SOLVERS = ("auto", "osqp", "scipy")


@dataclass
class RestrictedQPResult:
    x: np.ndarray
    support: np.ndarray
    objective: Optional[float]
    feasible: bool
    status: str
    solver: str
    solve_seconds: float
    iterations: int = 0
    constraint_dual: Optional[np.ndarray] = None
    maximum_violation: float = math.inf
    raw_status: Optional[str] = None
    infeasibility_certified: bool = False
    optimality_certified: bool = False


def _restricted_feasibility(
    instance: MarkowitzInstance,
    vector: np.ndarray,
    tolerance: float,
) -> tuple[bool, float]:
    if np.any(~np.isfinite(vector)):
        return False, math.inf
    values = np.asarray(instance.C @ vector, dtype=float).reshape(-1)
    lower = np.asarray(instance.lower, dtype=float)
    upper = np.asarray(instance.upper, dtype=float)
    lower_error = np.where(
        np.isfinite(lower),
        np.maximum(lower - values, 0.0),
        0.0,
    )
    upper_error = np.where(
        np.isfinite(upper),
        np.maximum(values - upper, 0.0),
        0.0,
    )
    maximum = max(
        max(0.0, -float(np.min(vector))),
        max(0.0, float(np.max(vector)) - 1.0),
        float(np.max(lower_error)) if lower_error.size else 0.0,
        float(np.max(upper_error)) if upper_error.size else 0.0,
    )
    return maximum <= float(tolerance), maximum


def _support(values: Sequence[int], dimension: int) -> np.ndarray:
    result = np.unique(np.asarray(values, dtype=np.int64).reshape(-1))
    if np.any(result < 0) or np.any(result >= dimension):
        raise ValueError("support contains an out-of-range index")
    return result


def _mapped_warm_start(
    warm_start: Optional[Any],
    support: np.ndarray,
    dimension: int,
) -> Optional[np.ndarray]:
    if warm_start is None:
        return None
    vector = np.asarray(warm_start, dtype=float).reshape(-1)
    if vector.shape == (dimension,):
        vector = vector[support]
    elif vector.shape != (support.size,):
        raise ValueError("restricted-QP warm start has the wrong dimension")
    return np.clip(vector, 0.0, 1.0)


def _empty_support_result(
    instance: MarkowitzInstance,
    support: np.ndarray,
    feasibility_tolerance: float,
) -> RestrictedQPResult:
    vector = np.zeros(instance.dimension)
    feasible, maximum = _restricted_feasibility(
        instance,
        vector,
        feasibility_tolerance,
    )
    return RestrictedQPResult(
        x=vector,
        support=support,
        objective=(
            sparse_objective(instance, vector) if feasible else None
        ),
        feasible=feasible,
        status=("solved" if feasible else "infeasible"),
        solver="analytic",
        solve_seconds=0.0,
        maximum_violation=maximum,
        infeasibility_certified=not feasible,
        optimality_certified=feasible,
    )


def _osqp_solve(
    instance: MarkowitzInstance,
    support: np.ndarray,
    warm_start: Optional[np.ndarray],
    settings: Mapping[str, Any],
) -> RestrictedQPResult:
    import osqp

    start = time.perf_counter()
    size = support.size
    rank = instance.rank
    rows = instance.rows
    factor = np.asarray(instance.B[support, :], dtype=float)
    constraint = sparse.csc_matrix(instance.C[:, support], dtype=float)
    variable_count = size + rank
    diagonal = np.concatenate(
        (
            np.full(size, float(instance.perspective_weight)),
            np.ones(rank),
        )
    )
    quadratic = sparse.diags(diagonal, format="csc")
    linear = np.concatenate(
        (
            -float(instance.return_reward)
            * np.asarray(instance.mu[support], dtype=float),
            np.zeros(rank),
        )
    )
    box_rows = sparse.hstack(
        (
            sparse.eye(size, format="csc"),
            sparse.csc_matrix((size, rank)),
        ),
        format="csc",
    )
    portfolio_rows = sparse.hstack(
        (constraint, sparse.csc_matrix((rows, rank))),
        format="csc",
    )
    factor_rows = sparse.hstack(
        (
            -sparse.csc_matrix(factor.T),
            sparse.eye(rank, format="csc"),
        ),
        format="csc",
    )
    matrix = sparse.vstack(
        (box_rows, portfolio_rows, factor_rows),
        format="csc",
    )
    lower = np.concatenate(
        (
            np.zeros(size),
            np.asarray(instance.lower, dtype=float),
            np.zeros(rank),
        )
    )
    upper = np.concatenate(
        (
            np.ones(size),
            np.asarray(instance.upper, dtype=float),
            np.zeros(rank),
        )
    )
    solver = osqp.OSQP()
    options = {
        "verbose": bool(settings.get("verbose", False)),
        "eps_abs": float(settings.get("eps_abs", 1e-9)),
        "eps_rel": float(settings.get("eps_rel", 1e-9)),
        "max_iter": int(settings.get("max_iterations", 20000)),
        "polishing": bool(settings.get("polishing", True)),
        "adaptive_rho": bool(settings.get("adaptive_rho", True)),
        "check_termination": int(settings.get("check_termination", 25)),
        "scaled_termination": bool(
            settings.get("scaled_termination", False)
        ),
    }
    time_limit = settings.get("time_limit")
    if time_limit is not None:
        options["time_limit"] = max(float(time_limit), 1e-6)
    solver.setup(
        P=quadratic,
        q=linear,
        A=matrix,
        l=lower,
        u=upper,
        **options,
    )
    if warm_start is not None:
        exposure = factor.T @ warm_start
        solver.warm_start(x=np.concatenate((warm_start, exposure)))
    result = solver.solve(raise_error=False)
    elapsed = time.perf_counter() - start
    raw_status = str(result.info.status)
    status_value = int(result.info.status_val)
    solved = status_value in {1, 2}
    full = np.zeros(instance.dimension)
    if result.x is not None and np.all(np.isfinite(result.x[:size])):
        full[support] = np.clip(result.x[:size], 0.0, 1.0)
    restricted_feasible, maximum_violation = _restricted_feasibility(
        instance,
        full,
        float(settings.get("feasibility_tolerance", 1e-7)),
    )
    # A feasible point is a valid incumbent even when OSQP stopped at a
    # time or iteration limit.  Only the separate certification flags may
    # close a BnB leaf.
    feasible = bool(restricted_feasible and result.x is not None)
    dual = None
    if result.y is not None and len(result.y) >= size + rows:
        dual = np.asarray(result.y[size : size + rows], dtype=float).copy()
    return RestrictedQPResult(
        x=full,
        support=support,
        objective=(sparse_objective(instance, full) if feasible else None),
        feasible=feasible,
        status=(
            "solved"
            if feasible and solved
            else (
                "feasible_not_converged"
                if feasible
                else raw_status.lower().replace(" ", "_")
            )
        ),
        solver="osqp",
        solve_seconds=elapsed,
        iterations=int(result.info.iter),
        constraint_dual=dual,
        maximum_violation=maximum_violation,
        raw_status=raw_status,
        # OSQP distinguishes its primal-infeasibility certificate from the
        # explicitly inaccurate variant. Only the former may close a BnB leaf.
        infeasibility_certified=status_value == 3,
        optimality_certified=bool(feasible and status_value == 1),
    )


def _linear_program_start(
    constraint: sparse.csc_matrix,
    lower: np.ndarray,
    upper: np.ndarray,
    size: int,
    time_limit: Optional[float],
) -> tuple[Optional[np.ndarray], str, bool]:
    equal = (
        np.isfinite(lower)
        & np.isfinite(upper)
        & (lower == upper)
    )
    upper_rows = np.flatnonzero(np.isfinite(upper) & ~equal)
    lower_rows = np.flatnonzero(np.isfinite(lower) & ~equal)
    inequality_parts = []
    inequality_bounds = []
    if upper_rows.size:
        inequality_parts.append(constraint[upper_rows, :])
        inequality_bounds.append(upper[upper_rows])
    if lower_rows.size:
        inequality_parts.append(-constraint[lower_rows, :])
        inequality_bounds.append(-lower[lower_rows])
    inequality = (
        sparse.vstack(inequality_parts, format="csc")
        if inequality_parts
        else None
    )
    inequality_bound = (
        np.concatenate(inequality_bounds) if inequality_bounds else None
    )
    equality_rows = np.flatnonzero(equal)
    equality = constraint[equality_rows, :] if equality_rows.size else None
    equality_bound = (
        0.5 * (lower[equality_rows] + upper[equality_rows])
        if equality_rows.size
        else None
    )
    options: dict[str, Any] = {}
    if time_limit is not None:
        options["time_limit"] = max(float(time_limit), 1e-6)
    result = optimize.linprog(
        np.zeros(size),
        A_ub=inequality,
        b_ub=inequality_bound,
        A_eq=equality,
        b_eq=equality_bound,
        bounds=(0.0, 1.0),
        method="highs",
        options=options,
    )
    if result.success and result.x is not None:
        return np.asarray(result.x, dtype=float), str(result.message), False
    return None, str(result.message), int(result.status) == 2


def _scipy_solve(
    instance: MarkowitzInstance,
    support: np.ndarray,
    warm_start: Optional[np.ndarray],
    settings: Mapping[str, Any],
) -> RestrictedQPResult:
    start = time.perf_counter()
    size = support.size
    factor = np.asarray(instance.B[support, :], dtype=float)
    constraint = sparse.csc_matrix(instance.C[:, support], dtype=float)
    lower = np.asarray(instance.lower, dtype=float)
    upper = np.asarray(instance.upper, dtype=float)
    tolerance = float(settings.get("feasibility_tolerance", 1e-7))
    time_limit = settings.get("time_limit")

    feasible_start = None
    if warm_start is not None:
        trial = np.zeros(instance.dimension)
        trial[support] = warm_start
        restricted_feasible, _ = _restricted_feasibility(
            instance,
            trial,
            tolerance,
        )
        if restricted_feasible:
            feasible_start = warm_start.copy()
    lp_status = "warm_start"
    lp_infeasible = False
    if feasible_start is None:
        feasible_start, lp_status, lp_infeasible = _linear_program_start(
            constraint,
            lower,
            upper,
            size,
            time_limit,
        )
    if feasible_start is None:
        return RestrictedQPResult(
            x=np.zeros(instance.dimension),
            support=support,
            objective=None,
            feasible=False,
            status=("infeasible" if lp_infeasible else "lp_failure"),
            solver="scipy",
            solve_seconds=time.perf_counter() - start,
            raw_status=lp_status,
            infeasibility_certified=lp_infeasible,
        )

    reward = float(instance.return_reward) * np.asarray(
        instance.mu[support],
        dtype=float,
    )
    ridge = float(instance.perspective_weight)

    def objective(vector: np.ndarray) -> float:
        exposure = factor.T @ vector
        return (
            0.5 * float(exposure @ exposure)
            + 0.5 * ridge * float(vector @ vector)
            - float(reward @ vector)
        )

    def gradient(vector: np.ndarray) -> np.ndarray:
        return factor @ (factor.T @ vector) + ridge * vector - reward

    constraints = []
    if instance.rows:
        equal = (
            np.isfinite(lower)
            & np.isfinite(upper)
            & (lower == upper)
        )
        equality_rows = np.flatnonzero(equal)
        interval_rows = np.flatnonzero(~equal)
        if equality_rows.size:
            equality_value = 0.5 * (
                lower[equality_rows] + upper[equality_rows]
            )
            constraints.append(
                optimize.LinearConstraint(
                    constraint[equality_rows, :],
                    equality_value,
                    equality_value,
                )
            )
        if interval_rows.size:
            constraints.append(
                optimize.LinearConstraint(
                    constraint[interval_rows, :],
                    lower[interval_rows],
                    upper[interval_rows],
                )
            )
    candidate = feasible_start
    raw_status = lp_status
    iterations = 0
    optimizer_converged = False
    try:
        result = optimize.minimize(
            objective,
            feasible_start,
            jac=gradient,
            method="SLSQP",
            bounds=optimize.Bounds(np.zeros(size), np.ones(size)),
            constraints=constraints,
            options={
                "maxiter": int(settings.get("max_iterations", 2000)),
                "ftol": float(settings.get("ftol", 1e-11)),
                "disp": bool(settings.get("verbose", False)),
            },
        )
        iterations = int(getattr(result, "nit", 0))
        raw_status = str(result.message)
        if result.x is not None and np.all(np.isfinite(result.x)):
            proposed = np.clip(np.asarray(result.x, dtype=float), 0.0, 1.0)
            full_proposed = np.zeros(instance.dimension)
            full_proposed[support] = proposed
            proposed_feasible, _ = _restricted_feasibility(
                instance,
                full_proposed,
                tolerance,
            )
            if (
                proposed_feasible
                and objective(proposed) <= objective(candidate) + 1e-12
            ):
                candidate = proposed
                optimizer_converged = bool(result.success)
    except (ValueError, RuntimeError) as error:
        raw_status = f"SLSQP fallback: {error}"

    full = np.zeros(instance.dimension)
    full[support] = candidate
    feasible, maximum_violation = _restricted_feasibility(
        instance,
        full,
        tolerance,
    )
    return RestrictedQPResult(
        x=full,
        support=support,
        objective=(sparse_objective(instance, full) if feasible else None),
        feasible=feasible,
        status=(
            "solved"
            if feasible and optimizer_converged
            else (
                "feasible_not_converged"
                if feasible
                else "numerical_failure"
            )
        ),
        solver="scipy",
        solve_seconds=time.perf_counter() - start,
        iterations=iterations,
        maximum_violation=maximum_violation,
        raw_status=raw_status,
        optimality_certified=bool(feasible and optimizer_converged),
    )


def solve_restricted_qp(
    instance: MarkowitzInstance,
    support: Sequence[int],
    *,
    solver: str = "auto",
    warm_start: Optional[Any] = None,
    options: Optional[Mapping[str, Any]] = None,
) -> RestrictedQPResult:
    """Solve the strongly convex portfolio QP on one proposed support."""
    if solver not in RESTRICTED_SOLVERS:
        raise ValueError(
            "restricted solver must be one of " + ", ".join(RESTRICTED_SOLVERS)
        )
    instance.validate()
    selected = _support(support, instance.dimension)
    settings = dict(options or {})
    tolerance = float(settings.get("feasibility_tolerance", 1e-7))
    if selected.size == 0:
        return _empty_support_result(instance, selected, tolerance)
    mapped = _mapped_warm_start(warm_start, selected, instance.dimension)
    actual_solver = solver
    if solver == "auto":
        actual_solver = (
            "osqp" if importlib.util.find_spec("osqp") is not None else "scipy"
        )
    if actual_solver == "osqp":
        if importlib.util.find_spec("osqp") is None:
            raise ImportError(
                "OSQP is not installed; install sksfolio[qp] or select scipy"
            )
        if solver == "osqp":
            return _osqp_solve(instance, selected, mapped, settings)
        try:
            return _osqp_solve(instance, selected, mapped, settings)
        except Exception as error:
            fallback = _scipy_solve(instance, selected, mapped, settings)
            prefix = f"OSQP runtime fallback ({type(error).__name__}: {error})"
            fallback.raw_status = (
                prefix
                if not fallback.raw_status
                else f"{prefix}; {fallback.raw_status}"
            )
            return fallback
    return _scipy_solve(instance, selected, mapped, settings)


__all__ = [
    "RESTRICTED_SOLVERS",
    "RestrictedQPResult",
    "solve_restricted_qp",
]
