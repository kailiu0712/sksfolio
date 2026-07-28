"""Proximal oracle for the perspective function and interval rows.

This module solves

    minimize_x  0.5 * ||x - v||^2 + gamma * G_k(x)
    subject to  lower <= C @ x <= upper.

For the exact budget row ``C = 1.T`` and ``lower = upper = 1``, the
specialized scalar Brent/PAVA oracle is used unchanged.  Otherwise the
Fenchel dual is solved by warm-started accelerated proximal gradient:

    minimize_y  h^*(-C.T @ y) + support_[lower, upper](y),

where ``h(x) = 0.5 * ||x - v||^2 + gamma * G_k(x)`` and

    grad h^*(-C.T @ y) = -C @ prox_{gamma G_k}(v - C.T @ y).

Thus, a general inner gradient evaluation requires one PAVA call and one
application each of ``C`` and ``C.T``.  A local-curvature line search avoids
the very conservative global step when ``gamma`` is large.  Constraint rows
are normalized internally.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Optional

import numpy as np
from scipy import sparse

from ..pdhg.pava import prox as pava_prox
from .budget_prox import prox_budget_details


_DENSE_OPERATOR_MIN_DENSITY = 0.20
_DENSE_OPERATOR_MAX_ENTRIES = 2_000_000


@dataclass(frozen=True)
class LinearProxResult:
    """Primal point and multiplier returned by a constrained prox solve."""

    x: np.ndarray
    constraint_multiplier: np.ndarray
    converged: bool
    iterations: int
    pava_calls: int
    fixed_point_residual: float
    constraint_violation: float
    scaled_constraint_violation: float
    method: str
    lipschitz: float
    restarts: int
    line_search_backtracks: int
    warm_started: bool


def _normalize_pava_method(method: str) -> str:
    normalized = str(method).lower().replace("-", "_")
    normalized = {
        "full": "full_sort",
        "partial": "partial_sort",
        "topk": "partial_sort",
    }.get(normalized, normalized)
    if normalized not in {"full_sort", "partial_sort"}:
        raise ValueError(
            "PAVA method must be 'full_sort' or 'partial_sort'"
        )
    return normalized


def _exact_budget_row(
    matrix: sparse.csr_matrix,
    lower: np.ndarray,
    upper: np.ndarray,
) -> bool:
    dimension = matrix.shape[1]
    return bool(
        matrix.shape == (1, dimension)
        and matrix.nnz == dimension
        and np.array_equal(matrix.indices, np.arange(dimension))
        and np.array_equal(matrix.data, np.ones(dimension))
        and lower.shape == (1,)
        and upper.shape == (1,)
        and lower[0] == 1.0
        and upper[0] == 1.0
    )


def _interval_projection(
    value: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    return np.minimum(np.maximum(value, lower), upper)


def _support_prox(
    value: np.ndarray,
    step: float,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Apply Moreau's identity to the support of an interval box."""
    projection = _interval_projection(value / step, lower, upper)
    return value - step * projection


def _interval_violation(
    value: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    if value.size == 0:
        return 0.0
    finite_lower = np.isfinite(lower)
    finite_upper = np.isfinite(upper)
    violation = 0.0
    if np.any(finite_lower):
        violation = max(
            violation,
            float(np.max(lower[finite_lower] - value[finite_lower])),
        )
    if np.any(finite_upper):
        violation = max(
            violation,
            float(np.max(value[finite_upper] - upper[finite_upper])),
        )
    return max(violation, 0.0)


def _scaled_perspective_conjugate(
    argument: np.ndarray,
    gamma: float,
    k: int,
) -> float:
    """Evaluate ``(gamma * G_k)^*`` in expected linear time."""
    penalties = np.zeros_like(argument)
    quadratic = (argument > 0.0) & (argument < gamma)
    linear = argument >= gamma
    penalties[quadratic] = (
        0.5
        * argument[quadratic]
        * (argument[quadratic] / gamma)
    )
    penalties[linear] = argument[linear] - 0.5 * gamma
    if k >= penalties.size:
        return float(np.sum(penalties))
    split = penalties.size - k
    return float(np.sum(np.partition(penalties, split)[split:]))


def _smooth_dual_value(
    shifted_argument: np.ndarray,
    point: np.ndarray,
    gamma: float,
    k: int,
) -> float:
    """Evaluate the smooth dual term whose gradient is the prox point."""
    residual = shifted_argument - point
    return (
        0.5 * float(point @ point)
        + _scaled_perspective_conjugate(residual, gamma, k)
    )


class LinearConstraintProx:
    """Reusable constrained perspective prox with dual warm starts."""

    def __init__(
        self,
        matrix: Any,
        lower: Any,
        upper: Any,
        k: int,
        *,
        pava_method: str = "partial_sort",
        tolerance: float = 1e-10,
        max_iterations: int = 1000,
        adaptive_restart: bool = True,
        use_budget_fast_path: bool = True,
    ) -> None:
        constraint = sparse.csr_matrix(matrix, dtype=np.float64)
        constraint.sum_duplicates()
        constraint.eliminate_zeros()
        constraint.sort_indices()
        lower_array = np.asarray(lower, dtype=np.float64).reshape(-1)
        upper_array = np.asarray(upper, dtype=np.float64).reshape(-1)
        if constraint.shape[0] != lower_array.size:
            raise ValueError("lower has the wrong dimension")
        if constraint.shape[0] != upper_array.size:
            raise ValueError("upper has the wrong dimension")
        if np.any(np.isnan(lower_array)) or np.any(np.isnan(upper_array)):
            raise ValueError("constraint bounds cannot contain NaN")
        if np.any(lower_array > upper_array):
            raise ValueError("a lower bound exceeds its upper bound")
        if np.any(~np.isfinite(constraint.data)):
            raise ValueError("constraint matrix must be finite")
        if isinstance(k, (bool, np.bool_)) or int(k) != k:
            raise ValueError("k must be an integer")
        k = int(k)
        if not 1 <= k <= constraint.shape[1]:
            raise ValueError("k must lie in {1, ..., dimension}")
        tolerance = float(tolerance)
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be positive and finite")
        max_iterations = int(max_iterations)
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")

        self.matrix = constraint
        self.lower = lower_array
        self.upper = upper_array
        self.k = k
        self.pava_method = _normalize_pava_method(pava_method)
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.adaptive_restart = bool(adaptive_restart)
        self.dimension = constraint.shape[1]
        self.rows = constraint.shape[0]
        self.exact_budget = bool(
            use_budget_fast_path
            and _exact_budget_row(
                constraint,
                lower_array,
                upper_array,
            )
        )

        squared_norms = np.asarray(
            constraint.multiply(constraint).sum(axis=1),
            dtype=float,
        ).reshape(-1)
        row_norms = np.sqrt(np.maximum(squared_norms, 0.0))
        row_norms[row_norms == 0.0] = 1.0
        self.row_norms = row_norms
        inverse_norms = 1.0 / row_norms
        self.scaled_matrix = sparse.diags(inverse_norms) @ constraint
        self.scaled_matrix = sparse.csr_matrix(self.scaled_matrix)
        operator_entries = self.rows * self.dimension
        operator_density = (
            self.scaled_matrix.nnz / operator_entries
            if operator_entries
            else 0.0
        )
        self.dense_operator = bool(
            operator_entries <= _DENSE_OPERATOR_MAX_ENTRIES
            and operator_density >= _DENSE_OPERATOR_MIN_DENSITY
        )
        if self.dense_operator:
            self.scaled_operator = np.asfortranarray(
                self.scaled_matrix.toarray()
            )
            self.scaled_transpose = self.scaled_operator.T
            self.operator_storage = "dense"
        else:
            self.scaled_operator = self.scaled_matrix
            self.scaled_transpose = self.scaled_matrix.transpose().tocsc(
                copy=False,
            )
            self.operator_storage = "sparse"
        self.scaled_lower = lower_array * inverse_norms
        self.scaled_upper = upper_array * inverse_norms
        self._warm_dual = np.zeros(self.rows, dtype=float)
        self._has_warm_start = False
        self._warm_budget_eta: float | None = None
        self.lipschitz, self.lipschitz_kind = self._dual_lipschitz()

    @property
    def method(self) -> str:
        if self.rows == 0:
            return "unconstrained_pava"
        if self.exact_budget:
            return "exact_budget_scalar_brent"
        return "row_scaled_dual_fista"

    def reset(self) -> None:
        """Discard the multiplier warm start."""
        self._warm_dual.fill(0.0)
        self._has_warm_start = False
        self._warm_budget_eta = None

    def snapshot(self) -> tuple[np.ndarray, bool, float | None]:
        """Return a copy of the committed warm-start state."""
        return (
            self._warm_dual.copy(),
            bool(self._has_warm_start),
            self._warm_budget_eta,
        )

    def restore(
        self,
        state: tuple[np.ndarray, bool, float | None],
    ) -> None:
        """Restore a state, for example after a rejected outer trial."""
        dual, available, budget_eta = state
        dual = np.asarray(dual, dtype=float).reshape(-1)
        if dual.shape != (self.rows,):
            raise ValueError("prox state has the wrong dimension")
        self._warm_dual = dual.copy()
        self._has_warm_start = bool(available)
        self._warm_budget_eta = budget_eta

    def _dual_lipschitz(self) -> tuple[float, str]:
        if self.rows == 0 or self.scaled_matrix.nnz == 0:
            return 0.0, "zero_operator"
        if self.rows <= 256:
            gram = self.scaled_matrix @ self.scaled_matrix.T
            if sparse.issparse(gram):
                gram = gram.toarray()
            eigenvalue = float(
                np.linalg.eigvalsh(np.asarray(gram, dtype=float))[-1]
            )
            # Guard against a last-bit underestimate of the spectral norm.
            return max(eigenvalue * (1.0 + 1e-12), 1e-15), "exact_row_gram"
        frobenius_squared = float(
            np.sum(self.scaled_matrix.data * self.scaled_matrix.data)
        )
        absolute = abs(self.scaled_matrix)
        maximum_row_sum = float(
            np.max(np.asarray(absolute.sum(axis=1)).reshape(-1))
        )
        maximum_column_sum = float(
            np.max(np.asarray(absolute.sum(axis=0)).reshape(-1))
        )
        one_infinity_bound = maximum_row_sum * maximum_column_sum
        upper_bound = min(frobenius_squared, one_infinity_bound)
        return max(upper_bound, 1e-15), "matrix_norm_upper_bound"

    def _pava(self, argument: np.ndarray, gamma: float) -> np.ndarray:
        return pava_prox(
            argument,
            gamma,
            self.k,
            self.pava_method,
        )

    def solve(
        self,
        argument: Any,
        gamma: float,
        *,
        tolerance: Optional[float] = None,
        max_iterations: Optional[int] = None,
    ) -> LinearProxResult:
        """Compute one constrained proximal point."""
        values = np.asarray(argument, dtype=np.float64).reshape(-1)
        if values.shape != (self.dimension,):
            raise ValueError("argument has the wrong dimension")
        if np.any(~np.isfinite(values)):
            raise ValueError("argument must be finite")
        gamma = float(gamma)
        if not math.isfinite(gamma) or gamma <= 0.0:
            raise ValueError("gamma must be positive and finite")
        active_tolerance = (
            self.tolerance if tolerance is None else float(tolerance)
        )
        active_limit = (
            self.max_iterations
            if max_iterations is None
            else int(max_iterations)
        )
        if (
            not math.isfinite(active_tolerance)
            or active_tolerance <= 0.0
        ):
            raise ValueError("tolerance must be positive and finite")
        if active_limit < 1:
            raise ValueError("max_iterations must be positive")

        if self.rows == 0:
            point = self._pava(values, gamma)
            return LinearProxResult(
                x=point,
                constraint_multiplier=np.empty(0, dtype=float),
                converged=True,
                iterations=1,
                pava_calls=1,
                fixed_point_residual=0.0,
                constraint_violation=0.0,
                scaled_constraint_violation=0.0,
                method=self.method,
                lipschitz=0.0,
                restarts=0,
                line_search_backtracks=0,
                warm_started=False,
            )

        if self.exact_budget:
            result = prox_budget_details(
                values,
                gamma,
                self.k,
                method=self.pava_method,
                tolerance=active_tolerance,
                max_iterations=active_limit,
                initial_eta=self._warm_budget_eta,
            )
            self._warm_budget_eta = float(result.eta)
            return LinearProxResult(
                x=np.asarray(result.x, dtype=float).reshape(-1),
                constraint_multiplier=np.array(
                    [result.eta],
                    dtype=float,
                ),
                converged=result.equality_residual <= active_tolerance,
                iterations=max(1, int(result.evaluations)),
                pava_calls=int(result.evaluations),
                fixed_point_residual=float(result.equality_residual),
                constraint_violation=float(result.equality_residual),
                scaled_constraint_violation=float(result.equality_residual),
                method=self.method,
                lipschitz=0.0,
                restarts=0,
                line_search_backtracks=0,
                warm_started=bool(result.warm_started),
            )

        if self.lipschitz == 0.0:
            point = self._pava(values, gamma)
            return LinearProxResult(
                x=point,
                constraint_multiplier=np.zeros(self.rows, dtype=float),
                converged=True,
                iterations=1,
                pava_calls=1,
                fixed_point_residual=0.0,
                constraint_violation=0.0,
                scaled_constraint_violation=0.0,
                method=self.method,
                lipschitz=0.0,
                restarts=0,
                line_search_backtracks=0,
                warm_started=self._has_warm_start,
            )

        dual = self._warm_dual.copy()
        extrapolated = dual.copy()
        momentum = 1.0
        warm_started = self._has_warm_start
        pava_calls = 0
        restarts = 0
        line_search_backtracks = 0
        fixed_point_residual = math.inf
        constraint_violation = math.inf
        scaled_constraint_violation = math.inf
        final_point = np.zeros(self.dimension, dtype=float)
        final_dual = dual
        converged = False
        completed = 0
        local_lipschitz = max(
            self.lipschitz / (1.0 + gamma),
            self.lipschitz * 1e-12,
            1e-15,
        )

        for iteration in range(1, active_limit + 1):
            shift = np.asarray(
                self.scaled_transpose @ extrapolated,
                dtype=float,
            ).reshape(-1)
            shifted_argument = values - shift
            point = self._pava(shifted_argument, gamma)
            pava_calls += 1
            row_value = np.asarray(
                self.scaled_operator @ point,
                dtype=float,
            ).reshape(-1)
            smooth_value = _smooth_dual_value(
                shifted_argument,
                point,
                gamma,
                self.k,
            )
            trial_lipschitz = max(
                local_lipschitz / 1.5,
                self.lipschitz * 1e-12,
                1e-15,
            )
            for _ in range(60):
                step = 1.0 / trial_lipschitz
                candidate = _support_prox(
                    extrapolated + step * row_value,
                    step,
                    self.scaled_lower,
                    self.scaled_upper,
                )
                candidate_shift = np.asarray(
                    self.scaled_transpose @ candidate,
                    dtype=float,
                ).reshape(-1)
                candidate_argument = values - candidate_shift
                candidate_point = self._pava(
                    candidate_argument,
                    gamma,
                )
                pava_calls += 1
                candidate_smooth = _smooth_dual_value(
                    candidate_argument,
                    candidate_point,
                    gamma,
                    self.k,
                )
                difference = candidate - extrapolated
                model = (
                    smooth_value
                    - float(row_value @ difference)
                    + 0.5
                    * trial_lipschitz
                    * float(difference @ difference)
                )
                scale = max(
                    1.0,
                    abs(smooth_value),
                    abs(candidate_smooth),
                )
                if candidate_smooth <= model + 1e-12 * scale:
                    break
                trial_lipschitz *= 2.0
                line_search_backtracks += 1
            else:
                raise RuntimeError(
                    "dual FISTA prox line search failed"
                )
            local_lipschitz = trial_lipschitz
            candidate_rows = np.asarray(
                self.scaled_operator @ candidate_point,
                dtype=float,
            ).reshape(-1)
            next_dual = _support_prox(
                candidate + step * candidate_rows,
                step,
                self.scaled_lower,
                self.scaled_upper,
            )
            fixed_point_residual = float(
                np.linalg.norm(next_dual - candidate, ord=np.inf)
                / step
            )
            scaled_constraint_violation = _interval_violation(
                candidate_rows,
                self.scaled_lower,
                self.scaled_upper,
            )
            original_rows = candidate_rows * self.row_norms
            constraint_violation = _interval_violation(
                original_rows,
                self.lower,
                self.upper,
            )
            final_point = candidate_point
            final_dual = candidate
            completed = iteration
            if (
                fixed_point_residual <= active_tolerance
                and scaled_constraint_violation <= active_tolerance
            ):
                converged = True
                break

            restart_now = bool(
                self.adaptive_restart
                and float(
                    (extrapolated - candidate) @ (candidate - dual)
                )
                > 0.0
            )
            if restart_now:
                next_momentum = 1.0
                next_extrapolated = candidate.copy()
                restarts += 1
            else:
                next_momentum = 0.5 * (
                    1.0 + math.sqrt(1.0 + 4.0 * momentum * momentum)
                )
                next_extrapolated = candidate + (
                    (momentum - 1.0) / next_momentum
                ) * (candidate - dual)
            dual = candidate
            extrapolated = next_extrapolated
            momentum = next_momentum

        self._warm_dual = final_dual.copy()
        self._has_warm_start = True
        original_multiplier = final_dual / self.row_norms
        return LinearProxResult(
            x=final_point,
            constraint_multiplier=original_multiplier,
            converged=converged,
            iterations=int(completed),
            pava_calls=int(pava_calls),
            fixed_point_residual=float(fixed_point_residual),
            constraint_violation=float(constraint_violation),
            scaled_constraint_violation=float(
                scaled_constraint_violation
            ),
            method=self.method,
            lipschitz=float(local_lipschitz),
            restarts=int(restarts),
            line_search_backtracks=int(line_search_backtracks),
            warm_started=warm_started,
        )


def prox_linear_details(
    argument: Any,
    gamma: float,
    k: int,
    matrix: Any,
    lower: Any,
    upper: Any,
    method: str = "partial_sort",
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 1000,
    adaptive_restart: bool = True,
) -> LinearProxResult:
    """One-shot constrained perspective prox with diagnostics."""
    oracle = LinearConstraintProx(
        matrix,
        lower,
        upper,
        k,
        pava_method=method,
        tolerance=tolerance,
        max_iterations=max_iterations,
        adaptive_restart=adaptive_restart,
    )
    return oracle.solve(argument, gamma)


def prox_linear(
    argument: Any,
    gamma: float,
    k: int,
    matrix: Any,
    lower: Any,
    upper: Any,
    method: str = "partial_sort",
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 1000,
    adaptive_restart: bool = True,
) -> np.ndarray:
    """Return only the constrained proximal point."""
    return prox_linear_details(
        argument,
        gamma,
        k,
        matrix,
        lower,
        upper,
        method,
        tolerance=tolerance,
        max_iterations=max_iterations,
        adaptive_restart=adaptive_restart,
    ).x


__all__ = [
    "LinearConstraintProx",
    "LinearProxResult",
    "prox_linear",
    "prox_linear_details",
]
