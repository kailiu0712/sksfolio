"""Large-scale PDHG solvers for the perspective Markowitz relaxation.

The implementation uses the saddle representation

    min_x max_{p,q} omega * g_k(x) - rho * mu.T @ x
        + <B.T @ x, p> - 0.5 * ||p||^2
        + <C_s @ x, q> - sigma_[lower_s, upper_s](q).

Here ``C_s`` and its interval bounds are scaled row by row.  Consequently,
each iteration needs only products with ``B`` and sparse ``C`` plus one PAVA
proximal call.  The dense covariance ``B @ B.T`` is never formed.

Five benchmark variants are exposed:

``fixed``
    Vanilla fixed-step Chambolle--Pock.
``fixed-restart``
    Fixed-step PDHG with practical residual restart.
``linesearch``
    Malitsky--Pock line-search PDHG.
``linesearch-restart``
    Line search plus practical residual restart.
``metric-linesearch-restart``
    The preceding method with the family calibration that was successful in
    the earlier large synthetic experiments.

The restart certificate is a fixed-reference proximal residual.  It is a
useful implementation heuristic, but it is not advertised here as the
normalized primal-dual-gap restart from a linear-rate theorem.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math
import time
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
from scipy import sparse

from ..problem import perspective_value as _perspective_value
from ..state import RelaxationState
from .pava import check_pava_oracles, prox as pava_prox
from .safe_dual import _dual_bound, evaluate_dual_bound


PDHG_VARIANTS = (
    "fixed",
    "fixed-restart",
    "linesearch",
    "linesearch-restart",
    "metric-linesearch-restart",
)

_DENSE_OPERATOR_MIN_DENSITY = 0.20
_DENSE_OPERATOR_MAX_ENTRIES = 2_000_000
_IMPLEMENTATION = (
    "native" if __name__.endswith("._native_solver") else "python"
)
_LANGUAGE = "cython" if _IMPLEMENTATION == "native" else "python"


@dataclass(frozen=True)
class _Settings:
    variant: str
    tolerance: float
    feasibility_tolerance: float
    time_limit: Optional[float]
    max_iterations: int
    check_interval: int
    min_epoch: int
    max_epoch: int
    restart_factor: float
    step_ratio: float
    constraint_weight: float
    reference_step_ratio: float
    reference_constraint_weight: float
    step_safety: float
    norm_iterations: int
    pava_backend: str
    line_search_delta: float
    line_search_shrink: float
    line_search_mode: str
    history_objective: bool
    dual_bound_cutoff: Optional[float]
    normalize_objective: bool
    center_return_objective: bool
    warm_start: Optional[RelaxationState]

    @property
    def line_search(self) -> bool:
        return "linesearch" in self.variant

    @property
    def restart(self) -> bool:
        return "restart" in self.variant


@dataclass(frozen=True)
class _ScaledProblem:
    B: Any
    mu: np.ndarray
    original_mu: np.ndarray
    C: Any
    lower: np.ndarray
    upper: np.ndarray
    original_C: Any
    original_lower: np.ndarray
    original_upper: np.ndarray
    anchor: np.ndarray
    k: int
    perspective_weight: float
    return_reward: float
    original_perspective_weight: float
    original_return_reward: float
    objective_scale: float
    factor_operator_scale: float
    return_centering_coefficients: np.ndarray
    return_centering_constant: float
    return_centering_equality_rows: np.ndarray
    return_centering_budget_coefficient: Optional[float]
    return_centering_implied_budget: Optional[float]
    return_centering_mu_mean: Optional[float]
    dimension: int
    factors: int
    constraints: int
    row_norms: np.ndarray
    constraint_operator_storage: str
    constraint_operator_density: float
    constraint_operator_entries: int
    constraint_operator_nnz: int
    constraint_names: Tuple[str, ...]


def _use_dense_constraint_operator(
    rows: int,
    dimension: int,
    nonzeros: int,
) -> bool:
    """Select dense BLAS only for moderate, sufficiently dense operators."""
    entries = int(rows) * int(dimension)
    density = float(nonzeros) / entries if entries else 0.0
    return bool(
        entries <= _DENSE_OPERATOR_MAX_ENTRIES
        and density >= _DENSE_OPERATOR_MIN_DENSITY
    )


def _as_options(options: Optional[Any]) -> Dict[str, Any]:
    if options is None:
        return {}
    if isinstance(options, Mapping):
        return dict(options)
    if hasattr(options, "__dict__"):
        return {
            key: value
            for key, value in vars(options).items()
            if not key.startswith("_")
        }
    raise TypeError("options must be a mapping, object, or None")


def _settings(
    dimension: int,
    variant: str,
    options: Optional[Any],
) -> _Settings:
    if variant not in PDHG_VARIANTS:
        raise ValueError(
            "variant must be one of " + ", ".join(PDHG_VARIANTS)
        )
    supplied = _as_options(options)
    tolerance = float(supplied.get("tolerance", 1e-6))
    feasibility_tolerance = float(
        supplied.get("feasibility_tolerance", tolerance)
    )
    raw_limit = supplied.get("time_limit")
    time_limit = None if raw_limit is None else float(raw_limit)
    max_iterations = int(supplied.get("max_iterations", 100_000))
    check_interval = int(supplied.get("check_interval", 50))
    min_epoch = int(supplied.get("min_epoch", 100))
    max_epoch = int(supplied.get("max_epoch", 2_000))
    restart_factor = float(supplied.get("restart_factor", 0.5))
    step_safety = float(supplied.get("step_safety", 0.98))
    norm_iterations = int(supplied.get("norm_iterations", 12))
    pava_backend = str(
        supplied.get("pava_backend", "partial_sort")
    ).lower().replace("-", "_")
    pava_backend = {
        "full": "full_sort",
        "partial": "partial_sort",
        "topk": "partial_sort",
    }.get(pava_backend, pava_backend)
    line_search_delta = float(
        supplied.get("line_search_delta", 0.999)
    )
    line_search_shrink = float(
        supplied.get("line_search_shrink", 0.9)
    )
    line_search_mode = str(
        supplied.get("line_search_mode", "auto")
    ).lower()
    history_objective = bool(
        supplied.get("history_objective", False)
    )
    normalize_objective = bool(
        supplied.get("normalize_objective", True)
    )
    center_return_objective = bool(
        supplied.get("center_return_objective", False)
    )
    raw_dual_bound_cutoff = supplied.get("dual_bound_cutoff")
    dual_bound_cutoff = (
        None
        if raw_dual_bound_cutoff is None
        else float(raw_dual_bound_cutoff)
    )
    warm_start = RelaxationState.coerce(
        supplied.get("warm_start", supplied.get("initial_state"))
    )

    base_weight = max(1.0, 0.0064 * float(dimension))
    metric_tuned = variant == "metric-linesearch-restart"
    default_ratio = 0.012 if metric_tuned else 0.032
    default_weight = 16.0 * base_weight if metric_tuned else base_weight
    step_ratio = float(supplied.get("step_ratio", default_ratio))
    constraint_weight = float(
        supplied.get("constraint_dual_weight", default_weight)
    )
    reference_step_ratio = float(
        supplied.get("reference_step_ratio", 0.032)
    )
    reference_constraint_weight = float(
        supplied.get(
            "reference_constraint_dual_weight",
            base_weight,
        )
    )

    positive_values = {
        "tolerance": tolerance,
        "feasibility_tolerance": feasibility_tolerance,
        "step_ratio": step_ratio,
        "constraint_dual_weight": constraint_weight,
        "reference_step_ratio": reference_step_ratio,
        "reference_constraint_dual_weight": (
            reference_constraint_weight
        ),
        "step_safety": step_safety,
    }
    for name, value in positive_values.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    if time_limit is not None and (
        not math.isfinite(time_limit) or time_limit <= 0.0
    ):
        raise ValueError("time_limit must be positive and finite")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if check_interval < 1:
        raise ValueError("check_interval must be positive")
    if min_epoch < check_interval:
        raise ValueError("min_epoch must be at least check_interval")
    if max_epoch < min_epoch:
        raise ValueError("max_epoch must be at least min_epoch")
    if not 0.0 < restart_factor < 1.0:
        raise ValueError("restart_factor must lie in (0, 1)")
    if not 0.0 < step_safety < 1.0:
        raise ValueError("step_safety must lie in (0, 1)")
    if norm_iterations < 1:
        raise ValueError("norm_iterations must be positive")
    if pava_backend not in {"partial_sort", "full_sort"}:
        raise ValueError(
            "pava_backend must be 'partial_sort' or 'full_sort'"
        )
    if not 0.0 < line_search_delta < 1.0:
        raise ValueError("line_search_delta must lie in (0, 1)")
    if not 0.0 < line_search_shrink < 1.0:
        raise ValueError("line_search_shrink must lie in (0, 1)")
    if line_search_mode not in {"auto", "gram", "direct"}:
        raise ValueError(
            "line_search_mode must be 'auto', 'gram', or 'direct'"
        )
    if dual_bound_cutoff is not None and not math.isfinite(
        dual_bound_cutoff
    ):
        raise ValueError("dual_bound_cutoff must be finite")
    return _Settings(
        variant=variant,
        tolerance=tolerance,
        feasibility_tolerance=feasibility_tolerance,
        time_limit=time_limit,
        max_iterations=max_iterations,
        check_interval=check_interval,
        min_epoch=min_epoch,
        max_epoch=max_epoch,
        restart_factor=restart_factor,
        step_ratio=step_ratio,
        constraint_weight=constraint_weight,
        reference_step_ratio=reference_step_ratio,
        reference_constraint_weight=reference_constraint_weight,
        step_safety=step_safety,
        norm_iterations=norm_iterations,
        pava_backend=pava_backend,
        line_search_delta=line_search_delta,
        line_search_shrink=line_search_shrink,
        line_search_mode=line_search_mode,
        history_objective=history_objective,
        dual_bound_cutoff=dual_bound_cutoff,
        normalize_objective=normalize_objective,
        center_return_objective=center_return_objective,
        warm_start=warm_start,
    )


def _prepare_problem(
    instance: Any,
    normalize_objective: bool = True,
    center_return_objective: bool = False,
) -> _ScaledProblem:
    original_B = instance.B
    original_perspective_weight = float(instance.perspective_weight)
    original_return_reward = float(instance.return_reward)
    if (
        not math.isfinite(original_perspective_weight)
        or original_perspective_weight <= 0.0
    ):
        raise ValueError(
            "perspective_weight must be positive and finite"
        )
    objective_scale = (
        original_perspective_weight
        if normalize_objective
        else 1.0
    )
    factor_operator_scale = 1.0 / math.sqrt(objective_scale)
    if (
        not math.isfinite(factor_operator_scale)
        or factor_operator_scale <= 0.0
    ):
        raise ValueError(
            "perspective_weight is too extreme for objective normalization"
        )
    B = original_B
    perspective_weight = (
        original_perspective_weight / objective_scale
    )
    return_reward = original_return_reward / objective_scale
    if not math.isfinite(return_reward):
        raise ValueError(
            "return_reward overflows under objective normalization"
        )
    if len(B.shape) != 2:
        raise ValueError("B must be a matrix")
    dimension, factors = map(int, B.shape)
    original_mu = np.asarray(
        instance.mu,
        dtype=float,
    ).reshape(-1)
    anchor = np.asarray(instance.anchor, dtype=float).reshape(-1).copy()
    if (
        original_mu.shape != (dimension,)
        or anchor.shape != (dimension,)
    ):
        raise ValueError("mu or anchor has the wrong dimension")

    original_C = instance.C
    C = sparse.csr_matrix(original_C, dtype=np.float64)
    C.sum_duplicates()
    C.eliminate_zeros()
    C.sort_indices()
    if C.shape[1] != dimension:
        raise ValueError("C has the wrong number of columns")
    constraints = int(C.shape[0])
    original_lower = np.asarray(instance.lower, dtype=float).reshape(-1)
    original_upper = np.asarray(instance.upper, dtype=float).reshape(-1)
    if (
        original_lower.shape != (constraints,)
        or original_upper.shape != (constraints,)
    ):
        raise ValueError("constraint bounds have the wrong dimension")
    if constraints:
        row_norms = np.sqrt(
            np.asarray(C.multiply(C).sum(axis=1)).reshape(-1)
        )
        if np.any(~np.isfinite(row_norms)) or np.any(row_norms <= 0.0):
            raise ValueError("every constraint row must be finite and nonzero")
        scaled_C = sparse.diags(1.0 / row_norms) @ C
        scaled_C = scaled_C.tocsr()
        lower = original_lower.copy()
        upper = original_upper.copy()
        finite_lower = np.isfinite(lower)
        finite_upper = np.isfinite(upper)
        lower[finite_lower] /= row_norms[finite_lower]
        upper[finite_upper] /= row_norms[finite_upper]
    else:
        row_norms = np.empty(0)
        scaled_C = C
        lower = original_lower.copy()
        upper = original_upper.copy()

    constraint_operator_entries = constraints * dimension
    constraint_operator_nnz = int(scaled_C.nnz)
    constraint_operator_density = (
        constraint_operator_nnz / constraint_operator_entries
        if constraint_operator_entries
        else 0.0
    )
    if _use_dense_constraint_operator(
        constraints,
        dimension,
        constraint_operator_nnz,
    ):
        operator_C: Any = np.asfortranarray(scaled_C.toarray())
        constraint_operator_storage = "dense"
    else:
        operator_C = scaled_C
        constraint_operator_storage = "sparse"

    mu = original_mu.copy()
    return_centering_coefficients = np.zeros(constraints)
    return_centering_constant = 0.0
    return_centering_equality_rows = np.empty(0, dtype=int)
    return_centering_budget_coefficient: Optional[float] = None
    return_centering_implied_budget: Optional[float] = None
    return_centering_mu_mean: Optional[float] = None
    if (
        center_return_objective
        and original_return_reward != 0.0
        and constraints
    ):
        equality_mask = (
            np.isfinite(original_lower)
            & np.isfinite(original_upper)
            & (original_lower == original_upper)
        )
        candidates: list[tuple[int, float]] = []
        for row_index in np.flatnonzero(equality_mask):
            start = int(C.indptr[row_index])
            stop = int(C.indptr[row_index + 1])
            row_data = C.data[start:stop]
            if row_data.size != dimension:
                continue
            coefficient = float(row_data[0])
            if (
                coefficient == 0.0
                or not np.all(row_data == coefficient)
            ):
                continue
            candidates.append((int(row_index), coefficient))

        if candidates:
            names = list(getattr(instance, "constraint_names", []))
            named = [
                candidate
                for candidate in candidates
                if candidate[0] < len(names)
                and names[candidate[0]].strip().lower()
                in {
                    "budget",
                    "full_investment",
                    "full-investment",
                }
            ]
            budget_row, budget_coefficient = (
                named[0] if named else candidates[0]
            )
            mu_mean = float(np.mean(original_mu))
            scaled_row_coefficient = (
                budget_coefficient / row_norms[budget_row]
            )
            row_scaled_centering_coefficient = (
                mu_mean / scaled_row_coefficient
            )
            implied_budget = (
                original_lower[budget_row] / budget_coefficient
            )
            if (
                not math.isfinite(row_scaled_centering_coefficient)
                or not math.isfinite(implied_budget)
            ):
                raise ValueError(
                    "budget return centering produced a nonfinite value"
                )
            mu -= mu_mean
            return_centering_coefficients[budget_row] = (
                row_scaled_centering_coefficient
            )
            return_centering_constant = (
                mu_mean * implied_budget
            )
            return_centering_equality_rows = np.array(
                [budget_row],
                dtype=int,
            )
            return_centering_budget_coefficient = budget_coefficient
            return_centering_implied_budget = float(implied_budget)
            return_centering_mu_mean = mu_mean

    return _ScaledProblem(
        B=B,
        mu=mu,
        original_mu=original_mu,
        C=operator_C,
        lower=lower,
        upper=upper,
        original_C=original_C,
        original_lower=original_lower,
        original_upper=original_upper,
        anchor=anchor,
        k=int(instance.k),
        perspective_weight=perspective_weight,
        return_reward=return_reward,
        original_perspective_weight=original_perspective_weight,
        original_return_reward=original_return_reward,
        objective_scale=objective_scale,
        factor_operator_scale=factor_operator_scale,
        return_centering_coefficients=(
            return_centering_coefficients
        ),
        return_centering_constant=return_centering_constant,
        return_centering_equality_rows=(
            return_centering_equality_rows
        ),
        return_centering_budget_coefficient=(
            return_centering_budget_coefficient
        ),
        return_centering_implied_budget=(
            return_centering_implied_budget
        ),
        return_centering_mu_mean=return_centering_mu_mean,
        dimension=dimension,
        factors=factors,
        constraints=constraints,
        row_norms=row_norms,
        constraint_operator_storage=constraint_operator_storage,
        constraint_operator_density=constraint_operator_density,
        constraint_operator_entries=constraint_operator_entries,
        constraint_operator_nnz=constraint_operator_nnz,
        constraint_names=tuple(
            str(name) for name in instance.constraint_names
        ),
    )


def _warm_start_point(
    problem: _ScaledProblem,
    settings: _Settings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, bool]:
    """Create a fresh PDHG epoch from a portable parent-node state."""
    state = settings.warm_start
    if state is None:
        return (
            problem.anchor.copy(),
            np.zeros(problem.factors),
            np.zeros(problem.constraints),
            False,
            False,
        )
    x = state.compatible_primal(problem)
    factor_dual = np.zeros(problem.factors)
    if (
        state.factor_dual is not None
        and state.factor_dual.shape == (problem.factors,)
    ):
        factor_dual = (
            state.factor_dual.copy()
            / math.sqrt(problem.objective_scale)
        )
    constraint_dual = np.zeros(problem.constraints)
    original_constraint = state.mapped_constraint_dual(problem)
    if original_constraint is not None:
        constraint_dual = (
            original_constraint * problem.row_norms
            - problem.original_return_reward
            * problem.return_centering_coefficients
        ) / problem.objective_scale
    exact_structure = state.constraint_ids == problem.constraint_names
    return x, factor_dual, constraint_dual, True, exact_structure


def _factor_forward(
    problem: _ScaledProblem,
    x: np.ndarray,
) -> np.ndarray:
    """Apply the objective-normalized factor operator."""
    return (
        problem.factor_operator_scale
        * np.asarray(problem.B.T @ x).reshape(-1)
    )


def _factor_adjoint(
    problem: _ScaledProblem,
    value: np.ndarray,
) -> np.ndarray:
    """Apply the adjoint objective-normalized factor operator."""
    return (
        problem.factor_operator_scale
        * np.asarray(problem.B @ value).reshape(-1)
    )


def _constraint_forward(
    problem: _ScaledProblem,
    x: np.ndarray,
) -> np.ndarray:
    """Apply the row-scaled interval-constraint operator."""
    return np.asarray(problem.C @ x).reshape(-1)


def _constraint_adjoint(
    problem: _ScaledProblem,
    value: np.ndarray,
) -> np.ndarray:
    """Apply the adjoint row-scaled interval-constraint operator."""
    return np.asarray(problem.C.T @ value).reshape(-1)


def _interval_dual_prox(
    value: np.ndarray,
    sigma: float,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    if value.size == 0:
        return value.copy()
    scaled = value / sigma
    projection = np.minimum(np.maximum(scaled, lower), upper)
    result = value - sigma * projection
    finite_lower = np.isfinite(lower)
    finite_upper = np.isfinite(upper)
    lower_only = finite_lower & ~finite_upper
    upper_only = ~finite_lower & finite_upper
    unbounded = ~finite_lower & ~finite_upper
    result[lower_only] = np.minimum(result[lower_only], 0.0)
    result[upper_only] = np.maximum(result[upper_only], 0.0)
    result[unbounded] = 0.0
    return result


def _operator_norm(
    problem: _ScaledProblem,
    constraint_weight: float,
    iterations: int,
) -> float:
    rng = np.random.default_rng(1381)
    vector = rng.normal(size=problem.dimension)
    vector /= max(float(np.linalg.norm(vector)), 1e-30)
    for _ in range(iterations):
        image = _factor_adjoint(
            problem,
            _factor_forward(problem, vector),
        )
        if problem.constraints:
            image = image + constraint_weight * _constraint_adjoint(
                problem,
                _constraint_forward(problem, vector),
            )
        image = np.asarray(image).reshape(-1)
        image_norm = float(np.linalg.norm(image))
        if image_norm == 0.0:
            return 0.0
        vector = image / image_norm
    image = _factor_adjoint(
        problem,
        _factor_forward(problem, vector),
    )
    if problem.constraints:
        image = image + constraint_weight * _constraint_adjoint(
            problem,
            _constraint_forward(problem, vector),
        )
    eigenvalue = float(vector @ np.asarray(image).reshape(-1))
    return math.sqrt(max(0.0, eigenvalue))


def _gram_data(
    problem: _ScaledProblem,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    scale = problem.factor_operator_scale
    factor_gram = scale * scale * np.asarray(
        problem.B.T @ problem.B
    )
    if problem.constraints:
        factor_constraint = (
            scale * np.asarray(problem.C @ problem.B).T
        )
        raw_constraint_gram = problem.C @ problem.C.T
        constraint_gram = (
            np.asarray(raw_constraint_gram.toarray())
            if sparse.issparse(raw_constraint_gram)
            else np.asarray(raw_constraint_gram)
        )
    else:
        factor_constraint = np.empty((problem.factors, 0))
        constraint_gram = np.empty((0, 0))
    return factor_gram, factor_constraint, constraint_gram


def _gram_norm(
    factor_gram: np.ndarray,
    factor_constraint: np.ndarray,
    constraint_gram: np.ndarray,
    constraint_weight: float,
) -> float:
    if constraint_gram.size == 0:
        eigenvalue = float(np.linalg.eigvalsh(factor_gram)[-1])
        return math.sqrt(max(0.0, eigenvalue))
    root_weight = math.sqrt(constraint_weight)
    transformed = np.block(
        [
            [
                factor_gram,
                root_weight * factor_constraint,
            ],
            [
                root_weight * factor_constraint.T,
                constraint_weight * constraint_gram,
            ],
        ]
    )
    eigenvalue = float(np.linalg.eigvalsh(transformed)[-1])
    return math.sqrt(max(0.0, eigenvalue))


def _frobenius_operator_bound(
    problem: _ScaledProblem,
    constraint_weight: float,
) -> float:
    if sparse.issparse(problem.B):
        factor_squared = float(
            np.asarray(problem.B.data) @ np.asarray(problem.B.data)
        )
    else:
        factor_squared = float(np.linalg.norm(problem.B) ** 2)
    factor_squared *= problem.factor_operator_scale**2
    if problem.constraints:
        if sparse.issparse(problem.C):
            constraint_squared = float(
                problem.C.data @ problem.C.data
            )
        else:
            constraint_squared = float(
                np.linalg.norm(problem.C) ** 2
            )
    else:
        constraint_squared = 0.0
    return math.sqrt(
        max(
            0.0,
            factor_squared
            + constraint_weight * constraint_squared,
        )
    )


def _maximum_violation(
    problem: _ScaledProblem,
    x: np.ndarray,
) -> float:
    if problem.constraints == 0:
        linear = 0.0
    else:
        values = np.asarray(problem.original_C @ x).reshape(-1)
        lower_error = np.where(
            np.isfinite(problem.original_lower),
            np.maximum(problem.original_lower - values, 0.0),
            0.0,
        )
        upper_error = np.where(
            np.isfinite(problem.original_upper),
            np.maximum(values - problem.original_upper, 0.0),
            0.0,
        )
        linear = max(
            float(np.max(lower_error)),
            float(np.max(upper_error)),
        )
    domain = max(
        0.0,
        -float(np.min(x)),
        float(np.max(x)) - 1.0,
        float(np.sum(x)) - float(problem.k),
    )
    return max(linear, domain)


def _objective(problem: _ScaledProblem, x: np.ndarray) -> float:
    exposure = np.asarray(problem.B.T @ x).reshape(-1)
    return (
        0.5 * float(exposure @ exposure)
        - problem.original_return_reward
        * float(problem.original_mu @ x)
        + problem.original_perspective_weight
        * _perspective_value(x, problem.k)
    )


def _residual(
    problem: _ScaledProblem,
    settings: _Settings,
    x: np.ndarray,
    factor_dual: np.ndarray,
    constraint_dual: np.ndarray,
    tau: float,
    sigma_factor: float,
    sigma_constraint: float,
) -> float:
    adjoint = _factor_adjoint(problem, factor_dual)
    if problem.constraints:
        adjoint = adjoint + _constraint_adjoint(
            problem,
            constraint_dual,
        )
    argument = (
        x
        - tau * np.asarray(adjoint).reshape(-1)
        + tau * problem.return_reward * problem.mu
    )
    primal_fixed = pava_prox(
        argument,
        tau * problem.perspective_weight,
        problem.k,
        settings.pava_backend,
    )
    factor_value = factor_dual + sigma_factor * (
        _factor_forward(problem, x)
    )
    factor_fixed = factor_value / (1.0 + sigma_factor)
    if problem.constraints:
        constraint_value = constraint_dual + sigma_constraint * (
            _constraint_forward(problem, x)
        )
        constraint_fixed = _interval_dual_prox(
            np.asarray(constraint_value).reshape(-1),
            sigma_constraint,
            problem.lower,
            problem.upper,
        )
        constraint_term = float(
            (constraint_dual - constraint_fixed)
            @ (constraint_dual - constraint_fixed)
        ) / sigma_constraint
    else:
        constraint_term = 0.0
    residual_squared = (
        float((x - primal_fixed) @ (x - primal_fixed)) / tau
        + float(
            (factor_dual - factor_fixed)
            @ (factor_dual - factor_fixed)
        )
        / sigma_factor
        + constraint_term
    )
    return math.sqrt(max(0.0, residual_squared))


def _history_point(
    problem: _ScaledProblem,
    settings: _Settings,
    iteration: int,
    elapsed: float,
    residual: float,
    residual_scale: float,
    violation: float,
    restart: bool,
    tau: float,
    backtracks: int,
    x: np.ndarray,
    dual_objective: float,
    best_dual_bound: float,
    dual_domain_correction: float,
) -> Dict[str, Any]:
    objective = (
        _objective(problem, x)
        if settings.history_objective
        else None
    )
    return {
        "iteration": int(iteration),
        "elapsed_seconds": float(elapsed),
        "residual": float(residual),
        "relative_residual": float(residual / residual_scale),
        "violation": float(violation),
        "objective": (
            float(objective)
            if objective is not None and math.isfinite(objective)
            else None
        ),
        "dual_objective": (
            float(dual_objective)
            if math.isfinite(dual_objective)
            else None
        ),
        "best_dual_bound": (
            float(best_dual_bound)
            if math.isfinite(best_dual_bound)
            else None
        ),
        "best_dual_lower_bound": (
            float(best_dual_bound)
            if math.isfinite(best_dual_bound)
            else None
        ),
        "dual_domain_correction": float(dual_domain_correction),
        "restart": bool(restart),
        "primal_step": float(tau),
        "cumulative_backtracks": int(backtracks),
    }


def _common_setup(
    problem: _ScaledProblem,
    settings: _Settings,
) -> Dict[str, Any]:
    gram_work = (
        problem.dimension
        * problem.factors
        * problem.factors
    )
    fixed_gram_affordable = (
        problem.factors + problem.constraints <= 256
        and gram_work <= 3.0e8
    )
    linesearch_gram_affordable = (
        problem.factors + problem.constraints <= 256
    )
    use_gram = False
    grams: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    if settings.line_search:
        use_gram = settings.line_search_mode == "gram" or (
            settings.line_search_mode == "auto"
            and linesearch_gram_affordable
        )
    else:
        # Fixed-step PDHG needs an upper bound, not a possibly low power
        # estimate.  The small Gram matrix is exact; when it is too costly,
        # the Frobenius norm is a rigorous (but potentially conservative)
        # upper bound.
        use_gram = fixed_gram_affordable
    if use_gram:
        grams = _gram_data(problem)
        norm_main = _gram_norm(
            *grams,
            settings.constraint_weight,
        )
        norm_reference = _gram_norm(
            *grams,
            settings.reference_constraint_weight,
        )
        norm_kind = "exact_dual_gram"
    elif not settings.line_search:
        norm_main = _frobenius_operator_bound(
            problem,
            settings.constraint_weight,
        )
        norm_reference = _frobenius_operator_bound(
            problem,
            settings.reference_constraint_weight,
        )
        norm_kind = "frobenius_upper_bound"
    else:
        norm_main = _operator_norm(
            problem,
            settings.constraint_weight,
            settings.norm_iterations,
        )
        if (
            settings.constraint_weight
            == settings.reference_constraint_weight
        ):
            norm_reference = norm_main
        else:
            norm_reference = _operator_norm(
                problem,
                settings.reference_constraint_weight,
                settings.norm_iterations,
            )
        norm_kind = "power_estimate_with_linesearch_safeguard"
    if norm_main <= 0.0 or norm_reference <= 0.0:
        raise RuntimeError("the stacked primal-dual operator is zero")

    tau = settings.step_safety / (
        settings.step_ratio * norm_main
    )
    alpha_factor = settings.step_ratio * settings.step_ratio
    alpha_constraint = alpha_factor * settings.constraint_weight
    reference_tau = settings.step_safety / (
        settings.reference_step_ratio * norm_reference
    )
    reference_alpha_factor = (
        settings.reference_step_ratio
        * settings.reference_step_ratio
    )
    reference_alpha_constraint = (
        reference_alpha_factor
        * settings.reference_constraint_weight
    )
    return {
        "use_gram": use_gram,
        "grams": grams,
        "operator_norm": norm_main,
        "reference_operator_norm": norm_reference,
        "operator_norm_kind": norm_kind,
        "fixed_step_stability_certified": (
            not settings.line_search
            and norm_kind
            in {"exact_dual_gram", "frobenius_upper_bound"}
        ),
        "tau": tau,
        "alpha_factor": alpha_factor,
        "alpha_constraint": alpha_constraint,
        "reference_tau": reference_tau,
        "reference_sigma_factor": (
            reference_tau * reference_alpha_factor
        ),
        "reference_sigma_constraint": (
            reference_tau * reference_alpha_constraint
        ),
    }


def _base_result(
    problem: _ScaledProblem,
    settings: _Settings,
    setup: Dict[str, Any],
    x: np.ndarray,
    status: str,
    iterations: int,
    residual: float,
    residual_scale: float,
    violation: float,
    pava_calls: int,
    restarts: int,
    backtracks: int,
    setup_seconds: float,
    solve_seconds: float,
    history: list[Dict[str, Any]],
    minimum_tau: float,
    maximum_tau: float,
    final_tau: float,
    best_dual_bound: float,
    best_dual_factor: np.ndarray,
    best_dual_constraint: np.ndarray,
    maximum_dual_domain_correction: float,
) -> Dict[str, Any]:
    original_factor_dual = (
        math.sqrt(problem.objective_scale) * best_dual_factor
    )
    original_scaled_constraint_dual = (
        problem.objective_scale * best_dual_constraint
        + problem.original_return_reward
        * problem.return_centering_coefficients
    )
    original_constraint_dual = (
        original_scaled_constraint_dual / problem.row_norms
        if problem.row_norms.size
        else original_scaled_constraint_dual.copy()
    )
    restart_state = RelaxationState(
        backend="pdhg",
        implementation=_IMPLEMENTATION,
        dimension=problem.dimension,
        k=problem.k,
        constraint_ids=problem.constraint_names,
        x=x,
        factor_dual=original_factor_dual,
        constraint_dual=original_constraint_dual,
        arrays={"extrapolated_x": x.copy()},
        scalars={
            "tau": float(final_tau),
            "fresh_restart_epoch_recommended": True,
        },
    )
    initial_dual_bound, _, _ = _dual_bound(
        problem,
        np.zeros(problem.factors),
        np.zeros(problem.constraints),
    )
    anchor_violation = _maximum_violation(problem, problem.anchor)
    anchor_objective = _objective(problem, problem.anchor)
    anchor_is_feasible = (
        math.isfinite(anchor_objective)
        and anchor_violation <= 1e-8
    )
    anchor_gap = (
        anchor_objective - best_dual_bound
        if anchor_is_feasible and math.isfinite(best_dual_bound)
        else None
    )
    relative_anchor_gap = (
        anchor_gap / max(1.0, abs(anchor_objective))
        if anchor_gap is not None
        else None
    )
    consistency_violation = (
        max(best_dual_bound - anchor_objective, 0.0)
        if anchor_is_feasible and math.isfinite(best_dual_bound)
        else None
    )
    return {
        "x": x,
        "status": status,
        "language": _LANGUAGE,
        "implementation": _IMPLEMENTATION,
        "native_solver_core": _IMPLEMENTATION == "native",
        "solver": "pdhg",
        "method": (
            "Malitsky--Pock PDHG"
            if settings.line_search
            else "Chambolle--Pock PDHG"
        ),
        "variant": settings.variant,
        "certified_optimal": False,
        "effective_tolerance": settings.tolerance,
        "effective_residual_tolerance": settings.tolerance,
        "feasibility_tolerance": settings.feasibility_tolerance,
        "stopping_certificate": (
            "initial_normalized_fixed_reference_kkt_residual"
            "_and_original_scale_feasibility"
        ),
        "objective_gap_certified": False,
        "dual_bound": (
            float(best_dual_bound)
            if math.isfinite(best_dual_bound)
            else None
        ),
        "best_dual_lower_bound": (
            float(best_dual_bound)
            if math.isfinite(best_dual_bound)
            else None
        ),
        "initial_dual_objective": (
            float(initial_dual_bound)
            if math.isfinite(initial_dual_bound)
            else None
        ),
        "last_evaluated_dual_objective": (
            history[-1].get("dual_objective")
            if history
            else (
                float(initial_dual_bound)
                if math.isfinite(initial_dual_bound)
                else None
            )
        ),
        "dual_bound_evaluations": int(1 + len(history)),
        "dual_bound_available": math.isfinite(best_dual_bound),
        "dual_bound_safe_in_exact_arithmetic": True,
        "dual_bound_floating_point_certified": False,
        "dual_bound_factor": original_factor_dual,
        "dual_bound_constraint_scaled": (
            original_scaled_constraint_dual
        ),
        "dual_bound_constraint_original": original_constraint_dual,
        "dual_bound_factor_internal": best_dual_factor.copy(),
        "dual_bound_constraint_internal": (
            best_dual_constraint.copy()
        ),
        "maximum_dual_domain_correction": float(
            maximum_dual_domain_correction
        ),
        "dual_domain_correction_coordinates": (
            "original_objective_row_scaled_constraints"
        ),
        "anchor_primal_upper_bound": (
            float(anchor_objective)
            if anchor_is_feasible
            else None
        ),
        "anchor_violation": float(anchor_violation),
        "anchor_is_numerically_feasible": bool(anchor_is_feasible),
        "anchor_primal_dual_gap": (
            float(anchor_gap) if anchor_gap is not None else None
        ),
        "relative_anchor_primal_dual_gap": (
            float(relative_anchor_gap)
            if relative_anchor_gap is not None
            else None
        ),
        "bound_consistency_violation": (
            float(consistency_violation)
            if consistency_violation is not None
            else None
        ),
        "dual_bound_kind": "fenchel_weak_duality_lower_bound",
        "dual_bound_units": "original_objective",
        "dual_multiplier_coordinates": {
            "dual_bound_factor": "original_objective_factor",
            "dual_bound_constraint_scaled": (
                "original_objective_row_scaled_constraints"
            ),
            "dual_bound_constraint_original": (
                "original_objective_original_constraint_rows"
            ),
            "dual_bound_factor_internal": (
                "objective_normalized_return_centered_factor"
            ),
            "dual_bound_constraint_internal": (
                "objective_normalized_return_centered_"
                "row_scaled_constraints"
            ),
        },
        "gap_kind": (
            "feasible_anchor_minus_running_best_fenchel_dual"
            if anchor_is_feasible
            else None
        ),
        "dual_bound_formula": (
            "-0.5*||p||^2 - support_[lower,upper](q) "
            "- (perspective_weight*G_k)^*("
            "return_reward*mu - B*p - C_scaled.T*q)"
        ),
        "dual_bound_formula_coordinates": (
            "original_objective_with_row_scaled_constraints"
        ),
        "dual_bound_cutoff": settings.dual_bound_cutoff,
        "objective_normalization_requested": bool(
            settings.normalize_objective
        ),
        "objective_normalization_applied": bool(
            settings.normalize_objective
            and problem.objective_scale != 1.0
        ),
        "objective_normalization_scale": float(
            problem.objective_scale
        ),
        "internal_perspective_weight": float(
            problem.perspective_weight
        ),
        "original_perspective_weight": float(
            problem.original_perspective_weight
        ),
        "internal_return_reward": float(problem.return_reward),
        "original_return_reward": float(
            problem.original_return_reward
        ),
        "return_centering_requested": bool(
            settings.center_return_objective
        ),
        "return_centering_applied": bool(
            np.any(problem.return_centering_coefficients != 0.0)
            and problem.original_return_reward != 0.0
        ),
        "return_centering_detection": (
            "exact_constant_coefficient_budget_equality"
        ),
        "return_centering_equality_rows": (
            problem.return_centering_equality_rows.copy()
        ),
        "return_centering_coefficients_row_scaled": (
            problem.return_centering_coefficients.copy()
        ),
        "return_centering_objective_constant": float(
            problem.return_centering_constant
        ),
        "return_centering_budget_coefficient": (
            problem.return_centering_budget_coefficient
        ),
        "return_centering_implied_budget": (
            problem.return_centering_implied_budget
        ),
        "return_centering_mu_mean": (
            problem.return_centering_mu_mean
        ),
        "return_centering_external_objective_offset": float(
            -problem.original_return_reward
            * problem.return_centering_constant
        ),
        "return_centering_original_mu_norm": float(
            np.linalg.norm(problem.original_mu)
        ),
        "return_centering_internal_mu_norm": float(
            np.linalg.norm(problem.mu)
        ),
        "iterations": int(iterations),
        "iteration_kind": "accepted_pdhg_iterations",
        "pava_calls": int(pava_calls),
        "pava_backend": settings.pava_backend,
        "restarts": int(restarts),
        "restart_certificate": (
            "fixed_reference_proximal_residual"
            if settings.restart
            else "disabled"
        ),
        "restart_action": (
            "extrapolation_reset"
            if settings.restart
            else "disabled"
        ),
        "line_search_backtracks": int(backtracks),
        "line_search_mode": (
            "gram" if setup["use_gram"] else "direct"
        ) if settings.line_search else "disabled",
        "residual": float(residual),
        "relative_residual": float(residual / residual_scale),
        "initial_residual": float(residual_scale),
        "residual_coordinates": (
            "objective_normalized_problem"
            if settings.normalize_objective
            else "original_objective_problem"
        ),
        "violation": float(violation),
        "setup_seconds": float(setup_seconds),
        "solve_seconds": float(solve_seconds),
        "total_seconds": float(setup_seconds + solve_seconds),
        "operator_norm": float(setup["operator_norm"]),
        "operator_norm_kind": setup["operator_norm_kind"],
        "fixed_step_stability_certified": setup[
            "fixed_step_stability_certified"
        ],
        "reference_operator_norm": float(
            setup["reference_operator_norm"]
        ),
        "step_ratio": settings.step_ratio,
        "constraint_dual_weight": settings.constraint_weight,
        "reference_step_ratio": settings.reference_step_ratio,
        "reference_constraint_dual_weight": (
            settings.reference_constraint_weight
        ),
        "initial_primal_step": float(setup["tau"]),
        "final_primal_step": float(final_tau),
        "minimum_primal_step": float(minimum_tau),
        "maximum_primal_step": float(maximum_tau),
        "history": history,
        "complexity_per_iteration": (
            (
                "O(d*r + d*m + PAVA(d,k))"
                if problem.constraint_operator_storage == "dense"
                else "O(d*r + nnz(C) + PAVA(d,k))"
            )
        ),
        "constraint_operator_storage": (
            problem.constraint_operator_storage
        ),
        "constraint_operator_density": float(
            problem.constraint_operator_density
        ),
        "constraint_operator_entries": int(
            problem.constraint_operator_entries
        ),
        "constraint_operator_nnz": int(
            problem.constraint_operator_nnz
        ),
        "constraint_operator_dense_min_density": float(
            _DENSE_OPERATOR_MIN_DENSITY
        ),
        "constraint_operator_dense_max_entries": int(
            _DENSE_OPERATOR_MAX_ENTRIES
        ),
        "row_scaling_min": (
            float(np.min(problem.row_norms))
            if problem.row_norms.size
            else None
        ),
        "row_scaling_max": (
            float(np.max(problem.row_norms))
            if problem.row_norms.size
            else None
        ),
        "constraint_row_normalization_applied": bool(
            problem.row_norms.size
        ),
        "constraint_row_normalization": (
            "divide_each_nonzero_row_and_its_finite_bounds_by_l2_norm"
        ),
        "warm_start_used": settings.warm_start is not None,
        "warm_start_exact_structure": bool(
            settings.warm_start is not None
            and settings.warm_start.constraint_ids
            == problem.constraint_names
        ),
        "warm_start_acceleration_resumed": False,
        "restart_state": restart_state.to_dict(copy=False),
    }


def _fixed_pdhg(
    problem: _ScaledProblem,
    settings: _Settings,
    setup: Dict[str, Any],
    setup_seconds: float,
) -> Dict[str, Any]:
    tau = float(setup["tau"])
    sigma_factor = tau * float(setup["alpha_factor"])
    sigma_constraint = tau * float(setup["alpha_constraint"])
    reference_tau = float(setup["reference_tau"])
    reference_sigma_factor = float(
        setup["reference_sigma_factor"]
    )
    reference_sigma_constraint = float(
        setup["reference_sigma_constraint"]
    )

    (
        x,
        factor_dual,
        constraint_dual,
        _,
        _,
    ) = _warm_start_point(problem, settings)
    extrapolated_x = x.copy()
    initial_residual = _residual(
        problem,
        settings,
        x,
        factor_dual,
        constraint_dual,
        reference_tau,
        reference_sigma_factor,
        reference_sigma_constraint,
    )
    residual_scale = max(initial_residual, 1e-16)
    best_x = x.copy()
    best_factor = factor_dual.copy()
    best_constraint = constraint_dual.copy()
    best_residual = initial_residual
    best_violation = _maximum_violation(problem, x)
    (
        best_dual_bound,
        best_dual_constraint,
        maximum_dual_domain_correction,
    ) = _dual_bound(
        problem,
        factor_dual,
        constraint_dual,
    )
    best_dual_factor = factor_dual.copy()
    epoch_best = (
        best_residual,
        best_x.copy(),
        best_factor.copy(),
        best_constraint.copy(),
    )
    anchor_residual = initial_residual
    epoch_start = 0
    pava_calls = 1
    restarts = 0
    history: list[Dict[str, Any]] = []
    status = "iteration_limit"
    completed = 0
    solve_start = time.perf_counter()

    for iteration in range(1, settings.max_iterations + 1):
        if (
            settings.time_limit is not None
            and time.perf_counter() - solve_start >= settings.time_limit
        ):
            status = "time_limit"
            break
        factor_value = factor_dual + sigma_factor * (
            _factor_forward(problem, extrapolated_x)
        )
        factor_next = factor_value / (1.0 + sigma_factor)
        if problem.constraints:
            constraint_value = (
                constraint_dual
                + sigma_constraint
                * _constraint_forward(problem, extrapolated_x)
            )
            constraint_next = _interval_dual_prox(
                np.asarray(constraint_value).reshape(-1),
                sigma_constraint,
                problem.lower,
                problem.upper,
            )
        else:
            constraint_next = constraint_dual.copy()
        adjoint = _factor_adjoint(problem, factor_next)
        if problem.constraints:
            adjoint = adjoint + _constraint_adjoint(
                problem,
                constraint_next,
            )
        argument = (
            x
            - tau * np.asarray(adjoint).reshape(-1)
            + tau * problem.return_reward * problem.mu
        )
        x_next = pava_prox(
            argument,
            tau * problem.perspective_weight,
            problem.k,
            settings.pava_backend,
        )
        pava_calls += 1
        extrapolated_next = 2.0 * x_next - x
        x = x_next
        factor_dual = factor_next
        constraint_dual = constraint_next
        extrapolated_x = extrapolated_next
        completed = iteration

        if (
            iteration % settings.check_interval != 0
            and iteration < settings.max_iterations
        ):
            continue
        elapsed = time.perf_counter() - solve_start
        residual = _residual(
            problem,
            settings,
            x,
            factor_dual,
            constraint_dual,
            reference_tau,
            reference_sigma_factor,
            reference_sigma_constraint,
        )
        pava_calls += 1
        violation = _maximum_violation(problem, x)
        if residual < epoch_best[0]:
            epoch_best = (
                residual,
                x.copy(),
                factor_dual.copy(),
                constraint_dual.copy(),
            )
        if (
            residual < best_residual
            or (
                residual <= best_residual * (1.0 + 1e-12)
                and violation < best_violation
            )
        ):
            best_x = x.copy()
            best_factor = factor_dual.copy()
            best_constraint = constraint_dual.copy()
            best_residual = residual
            best_violation = violation

        restart_now = False
        if settings.restart:
            epoch_length = iteration - epoch_start
            restart_now = (
                epoch_length >= settings.min_epoch
                and residual
                <= settings.restart_factor * anchor_residual
            )
            if epoch_length >= settings.max_epoch and not restart_now:
                (
                    residual,
                    x,
                    factor_dual,
                    constraint_dual,
                ) = (
                    epoch_best[0],
                    epoch_best[1].copy(),
                    epoch_best[2].copy(),
                    epoch_best[3].copy(),
                )
                violation = _maximum_violation(problem, x)
                restart_now = True

        (
            dual_objective,
            safe_constraint_dual,
            dual_domain_correction,
        ) = _dual_bound(
            problem,
            factor_dual,
            constraint_dual,
        )
        maximum_dual_domain_correction = max(
            maximum_dual_domain_correction,
            dual_domain_correction,
        )
        if dual_objective > best_dual_bound:
            best_dual_bound = dual_objective
            best_dual_factor = factor_dual.copy()
            best_dual_constraint = safe_constraint_dual.copy()

        history_point = _history_point(
            problem,
            settings,
            iteration,
            elapsed,
            residual,
            residual_scale,
            violation,
            restart_now,
            tau,
            0,
            x,
            dual_objective,
            best_dual_bound,
            dual_domain_correction,
        )
        elapsed = time.perf_counter() - solve_start
        history_point["elapsed_seconds"] = float(elapsed)
        history.append(history_point)
        if (
            settings.dual_bound_cutoff is not None
            and best_dual_bound >= settings.dual_bound_cutoff
        ):
            status = "dual_bound_cutoff"
            break
        if (
            residual / residual_scale <= settings.tolerance
            and violation <= settings.feasibility_tolerance
        ):
            best_x = x.copy()
            best_factor = factor_dual.copy()
            best_constraint = constraint_dual.copy()
            best_residual = residual
            best_violation = violation
            status = "converged"
            break
        if (
            settings.time_limit is not None
            and elapsed >= settings.time_limit
        ):
            status = "time_limit"
            break
        if restart_now:
            extrapolated_x = x.copy()
            anchor_residual = residual
            epoch_start = iteration
            epoch_best = (
                residual,
                x.copy(),
                factor_dual.copy(),
                constraint_dual.copy(),
            )
            restarts += 1

    solve_seconds = time.perf_counter() - solve_start
    return _base_result(
        problem,
        settings,
        setup,
        best_x,
        status,
        completed,
        best_residual,
        residual_scale,
        best_violation,
        pava_calls,
        restarts,
        0,
        setup_seconds,
        solve_seconds,
        history,
        tau,
        tau,
        tau,
        best_dual_bound,
        best_dual_factor,
        best_dual_constraint,
        maximum_dual_domain_correction,
    )


def _linesearch_pdhg(
    problem: _ScaledProblem,
    settings: _Settings,
    setup: Dict[str, Any],
    setup_seconds: float,
) -> Dict[str, Any]:
    tau = float(setup["tau"])
    alpha_factor = float(setup["alpha_factor"])
    alpha_constraint = float(setup["alpha_constraint"])
    reference_tau = float(setup["reference_tau"])
    reference_sigma_factor = float(
        setup["reference_sigma_factor"]
    )
    reference_sigma_constraint = float(
        setup["reference_sigma_constraint"]
    )
    grams = setup["grams"]

    (
        x,
        factor_dual,
        constraint_dual,
        _,
        _,
    ) = _warm_start_point(problem, settings)
    factor_image = _factor_forward(problem, x)
    constraint_image = _constraint_forward(problem, x)
    adjoint_dual = np.zeros(problem.dimension)
    initial_residual = _residual(
        problem,
        settings,
        x,
        factor_dual,
        constraint_dual,
        reference_tau,
        reference_sigma_factor,
        reference_sigma_constraint,
    )
    residual_scale = max(initial_residual, 1e-16)
    best_x = x.copy()
    best_factor = factor_dual.copy()
    best_constraint = constraint_dual.copy()
    best_residual = initial_residual
    best_violation = _maximum_violation(problem, x)
    (
        best_dual_bound,
        best_dual_constraint,
        maximum_dual_domain_correction,
    ) = _dual_bound(
        problem,
        factor_dual,
        constraint_dual,
    )
    best_dual_factor = factor_dual.copy()
    epoch_best = (
        best_residual,
        best_x.copy(),
        best_factor.copy(),
        best_constraint.copy(),
    )
    anchor_residual = initial_residual
    epoch_start = 0
    theta = 1.0
    minimum_tau = tau
    maximum_tau = tau
    backtracks = 0
    pava_calls = 1
    restarts = 0
    history: list[Dict[str, Any]] = []
    status = "iteration_limit"
    completed = 0
    solve_start = time.perf_counter()

    for iteration in range(1, settings.max_iterations + 1):
        if (
            settings.time_limit is not None
            and time.perf_counter() - solve_start >= settings.time_limit
        ):
            status = "time_limit"
            break
        argument = (
            x
            - tau * adjoint_dual
            + tau * problem.return_reward * problem.mu
        )
        x_next = pava_prox(
            argument,
            tau * problem.perspective_weight,
            problem.k,
            settings.pava_backend,
        )
        pava_calls += 1
        factor_image_next = _factor_forward(problem, x_next)
        constraint_image_next = _constraint_forward(problem, x_next)

        tau_trial = tau * math.sqrt(1.0 + theta)
        accepted = False
        for _ in range(80):
            theta_trial = tau_trial / tau
            extrapolated_factor = factor_image_next + theta_trial * (
                factor_image_next - factor_image
            )
            extrapolated_constraint = (
                constraint_image_next
                + theta_trial
                * (constraint_image_next - constraint_image)
            )
            sigma_factor = tau_trial * alpha_factor
            sigma_constraint = tau_trial * alpha_constraint
            factor_trial = (
                factor_dual + sigma_factor * extrapolated_factor
            ) / (1.0 + sigma_factor)
            if problem.constraints:
                constraint_value = (
                    constraint_dual
                    + sigma_constraint * extrapolated_constraint
                )
                constraint_trial = _interval_dual_prox(
                    constraint_value,
                    sigma_constraint,
                    problem.lower,
                    problem.upper,
                )
            else:
                constraint_trial = constraint_dual.copy()
            factor_difference = factor_trial - factor_dual
            constraint_difference = (
                constraint_trial - constraint_dual
            )

            if setup["use_gram"]:
                assert grams is not None
                factor_gram, factor_constraint, constraint_gram = grams
                adjoint_norm_squared = float(
                    factor_difference
                    @ factor_gram
                    @ factor_difference
                )
                if problem.constraints:
                    adjoint_norm_squared += float(
                        2.0
                        * factor_difference
                        @ factor_constraint
                        @ constraint_difference
                        + constraint_difference
                        @ constraint_gram
                        @ constraint_difference
                    )
                adjoint_norm_squared = max(
                    0.0,
                    adjoint_norm_squared,
                )
            else:
                adjoint_difference = _factor_adjoint(
                    problem,
                    factor_difference,
                )
                if problem.constraints:
                    adjoint_difference = (
                        adjoint_difference
                        + _constraint_adjoint(
                            problem,
                            constraint_difference,
                        )
                    )
                adjoint_difference = np.asarray(
                    adjoint_difference
                ).reshape(-1)
                adjoint_norm_squared = float(
                    adjoint_difference @ adjoint_difference
                )
            left = tau_trial * tau_trial * adjoint_norm_squared
            right = settings.line_search_delta**2 * (
                float(factor_difference @ factor_difference)
                / alpha_factor
                + (
                    float(
                        constraint_difference
                        @ constraint_difference
                    )
                    / alpha_constraint
                    if problem.constraints
                    else 0.0
                )
            )
            if (
                math.isfinite(left)
                and left <= right * (1.0 + 1e-12) + 1e-30
            ):
                accepted = True
                break
            tau_trial *= settings.line_search_shrink
            backtracks += 1
        if not accepted:
            raise RuntimeError(
                "PDHG line search failed after 80 backtracks"
            )
        if setup["use_gram"]:
            adjoint_difference = _factor_adjoint(
                problem,
                factor_difference,
            )
            if problem.constraints:
                adjoint_difference = (
                    adjoint_difference
                    + _constraint_adjoint(
                        problem,
                        constraint_difference,
                    )
                )
            adjoint_difference = np.asarray(
                adjoint_difference
            ).reshape(-1)

        x = x_next
        factor_image = factor_image_next
        constraint_image = constraint_image_next
        factor_dual = factor_trial
        constraint_dual = constraint_trial
        adjoint_dual += adjoint_difference
        tau = tau_trial
        theta = theta_trial
        minimum_tau = min(minimum_tau, tau)
        maximum_tau = max(maximum_tau, tau)
        completed = iteration

        if (
            iteration % settings.check_interval != 0
            and iteration < settings.max_iterations
        ):
            continue
        elapsed = time.perf_counter() - solve_start
        residual = _residual(
            problem,
            settings,
            x,
            factor_dual,
            constraint_dual,
            reference_tau,
            reference_sigma_factor,
            reference_sigma_constraint,
        )
        pava_calls += 1
        violation = _maximum_violation(problem, x)
        if residual < epoch_best[0]:
            epoch_best = (
                residual,
                x.copy(),
                factor_dual.copy(),
                constraint_dual.copy(),
            )
        if (
            residual < best_residual
            or (
                residual <= best_residual * (1.0 + 1e-12)
                and violation < best_violation
            )
        ):
            best_x = x.copy()
            best_factor = factor_dual.copy()
            best_constraint = constraint_dual.copy()
            best_residual = residual
            best_violation = violation

        restart_now = False
        if settings.restart:
            epoch_length = iteration - epoch_start
            restart_now = (
                epoch_length >= settings.min_epoch
                and residual
                <= settings.restart_factor * anchor_residual
            )
            if epoch_length >= settings.max_epoch and not restart_now:
                (
                    residual,
                    x,
                    factor_dual,
                    constraint_dual,
                ) = (
                    epoch_best[0],
                    epoch_best[1].copy(),
                    epoch_best[2].copy(),
                    epoch_best[3].copy(),
                )
                violation = _maximum_violation(problem, x)
                restart_now = True

        (
            dual_objective,
            safe_constraint_dual,
            dual_domain_correction,
        ) = _dual_bound(
            problem,
            factor_dual,
            constraint_dual,
        )
        maximum_dual_domain_correction = max(
            maximum_dual_domain_correction,
            dual_domain_correction,
        )
        if dual_objective > best_dual_bound:
            best_dual_bound = dual_objective
            best_dual_factor = factor_dual.copy()
            best_dual_constraint = safe_constraint_dual.copy()

        history_point = _history_point(
            problem,
            settings,
            iteration,
            elapsed,
            residual,
            residual_scale,
            violation,
            restart_now,
            tau,
            backtracks,
            x,
            dual_objective,
            best_dual_bound,
            dual_domain_correction,
        )
        elapsed = time.perf_counter() - solve_start
        history_point["elapsed_seconds"] = float(elapsed)
        history.append(history_point)
        if (
            settings.dual_bound_cutoff is not None
            and best_dual_bound >= settings.dual_bound_cutoff
        ):
            status = "dual_bound_cutoff"
            break
        if (
            residual / residual_scale <= settings.tolerance
            and violation <= settings.feasibility_tolerance
        ):
            best_x = x.copy()
            best_factor = factor_dual.copy()
            best_constraint = constraint_dual.copy()
            best_residual = residual
            best_violation = violation
            status = "converged"
            break
        if (
            settings.time_limit is not None
            and elapsed >= settings.time_limit
        ):
            status = "time_limit"
            break
        if restart_now:
            factor_image = _factor_forward(problem, x)
            constraint_image = _constraint_forward(problem, x)
            adjoint_dual = _factor_adjoint(
                problem,
                factor_dual,
            )
            if problem.constraints:
                adjoint_dual += _constraint_adjoint(
                    problem,
                    constraint_dual,
                )
            theta = 1.0
            anchor_residual = residual
            epoch_start = iteration
            epoch_best = (
                residual,
                x.copy(),
                factor_dual.copy(),
                constraint_dual.copy(),
            )
            restarts += 1

    solve_seconds = time.perf_counter() - solve_start
    return _base_result(
        problem,
        settings,
        setup,
        best_x,
        status,
        completed,
        best_residual,
        residual_scale,
        best_violation,
        pava_calls,
        restarts,
        backtracks,
        setup_seconds,
        solve_seconds,
        history,
        minimum_tau,
        maximum_tau,
        tau,
        best_dual_bound,
        best_dual_factor,
        best_dual_constraint,
        maximum_dual_domain_correction,
    )


def solve_pdhg(
    instance: Any,
    variant: str = "metric-linesearch-restart",
    options: Optional[Any] = None,
) -> Dict[str, Any]:
    """Solve one bundle-compatible instance with a selected PDHG variant."""
    supplied = _as_options(options)
    threads = int(supplied.get("threads", 0))
    if threads < 0:
        raise ValueError("threads must be nonnegative")
    if threads:
        try:
            from threadpoolctl import threadpool_info, threadpool_limits
        except ImportError as error:
            raise RuntimeError(
                "threadpoolctl is required to enforce PDHG thread limits"
            ) from error
        thread_context = threadpool_limits(
            limits=threads,
            user_api="blas",
        )
    else:
        thread_context = nullcontext()

    with thread_context:
        if threads:
            threadpools = threadpool_info()
        else:
            try:
                from threadpoolctl import threadpool_info

                threadpools = threadpool_info()
            except ImportError:
                threadpools = []
        total_start = time.perf_counter()
        raw_B = instance.B
        if len(raw_B.shape) != 2:
            raise ValueError("B must be a matrix")
        settings = _settings(int(raw_B.shape[0]), variant, supplied)
        problem = _prepare_problem(
            instance,
            normalize_objective=settings.normalize_objective,
            center_return_objective=(
                settings.center_return_objective
            ),
        )
        setup = _common_setup(problem, settings)
        setup_seconds = time.perf_counter() - total_start
        if settings.line_search:
            result = _linesearch_pdhg(
                problem,
                settings,
                setup,
                setup_seconds,
            )
        else:
            result = _fixed_pdhg(
                problem,
                settings,
                setup,
                setup_seconds,
            )
    result["threads"] = threads
    result["thread_limit_kind"] = (
        "blas" if threads else "runtime_default"
    )
    result["threadpools"] = threadpools
    return result


__all__ = [
    "PDHG_VARIANTS",
    "check_pava_oracles",
    "evaluate_dual_bound",
    "solve_pdhg",
]
