"""Proximal oracle for the perspective function and interval rows.

This module solves

    minimize_x  0.5 * ||x - v||^2 + gamma * G_k(x)
    subject to  lower <= C @ x <= upper.

For the exact budget row ``C = 1.T`` and ``lower = upper = 1``, the
specialized scalar Brent/PAVA oracle is used unchanged. Otherwise the Fenchel
dual can be solved by warm-started accelerated proximal gradient or by an
exact equality/upper/lower multiplier split followed by L-BFGS-B:

    minimize_y  h^*(-C.T @ y) + support_[lower, upper](y),

where ``h(x) = 0.5 * ||x - v||^2 + gamma * G_k(x)`` and

    grad h^*(-C.T @ y) = -C @ prox_{gamma G_k}(v - C.T @ y).

Thus, a general inner gradient evaluation requires one PAVA call and one
application each of ``C`` and ``C.T``. The dual-FISTA path uses a
local-curvature line search. The L-BFGS-B path independently verifies the
signed fixed-point residual and primal interval violation, then falls back to
dual FISTA if needed. Constraint rows are normalized internally.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Optional

import numpy as np
from scipy import sparse
from scipy.optimize import fmin_l_bfgs_b

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
    function_evaluations: int = 0
    fallback_used: bool = False
    optimizer_status: str = ""


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
        dual_solver: str = "lbfgs",
        lbfgs_memory: int = 10,
        lbfgs_max_line_search: int = 40,
        lbfgs_fallback: bool = True,
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
        dual_solver = str(dual_solver).lower().replace("-", "_")
        dual_solver = {
            "apg": "fista",
            "dual_fista": "fista",
            "l_bfgs": "lbfgs",
            "lbfgsb": "lbfgs",
            "l_bfgs_b": "lbfgs",
            "dual_lbfgs": "lbfgs",
        }.get(dual_solver, dual_solver)
        if dual_solver not in {"fista", "lbfgs"}:
            raise ValueError("dual_solver must be 'fista' or 'lbfgs'")
        lbfgs_memory = int(lbfgs_memory)
        lbfgs_max_line_search = int(lbfgs_max_line_search)
        if lbfgs_memory < 1:
            raise ValueError("lbfgs_memory must be positive")
        if lbfgs_max_line_search < 1:
            raise ValueError("lbfgs_max_line_search must be positive")

        self.matrix = constraint
        self.lower = lower_array
        self.upper = upper_array
        self.k = k
        self.pava_method = _normalize_pava_method(pava_method)
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.adaptive_restart = bool(adaptive_restart)
        self.dual_solver = dual_solver
        self.lbfgs_memory = lbfgs_memory
        self.lbfgs_max_line_search = lbfgs_max_line_search
        self.lbfgs_fallback = bool(lbfgs_fallback)
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

        finite_lower = np.isfinite(self.scaled_lower)
        finite_upper = np.isfinite(self.scaled_upper)
        equality = (
            finite_lower
            & finite_upper
            & (self.scaled_lower == self.scaled_upper)
        )
        self._lbfgs_equal = np.flatnonzero(equality)
        self._lbfgs_upper = np.flatnonzero(
            finite_upper & ~equality
        )
        self._lbfgs_lower = np.flatnonzero(
            finite_lower & ~equality
        )

    @property
    def method(self) -> str:
        if self.rows == 0:
            return "unconstrained_pava"
        if self.exact_budget:
            return "exact_budget_scalar_brent"
        if self.dual_solver == "lbfgs":
            return "row_scaled_dual_lbfgsb"
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

    def initialize_warm_start(
        self,
        *,
        original_multiplier: Optional[Any] = None,
        prox_step: float = 1.0,
        internal_dual: Optional[Any] = None,
        budget_eta: Optional[float] = None,
    ) -> None:
        """Initialize a reusable prox state from a previous solve.

        ``original_multiplier`` is expressed in the original constraint-row
        coordinates and in the unscaled outer objective.  The proximal dual
        is step-scaled, so a parent-node multiplier is multiplied by the new
        outer step before conversion to the row-normalized coordinates.
        ``internal_dual`` takes precedence and is intended for exact same-node
        continuation.
        """
        if internal_dual is not None:
            dual = np.asarray(internal_dual, dtype=float).reshape(-1)
        elif original_multiplier is not None:
            multiplier = np.asarray(
                original_multiplier,
                dtype=float,
            ).reshape(-1)
            if multiplier.shape != (self.rows,):
                raise ValueError(
                    "warm-start multiplier has the wrong dimension"
                )
            prox_step = float(prox_step)
            if not math.isfinite(prox_step) or prox_step <= 0.0:
                raise ValueError("prox_step must be positive and finite")
            dual = multiplier * prox_step * self.row_norms
        else:
            dual = np.zeros(self.rows, dtype=float)
        if dual.shape != (self.rows,):
            raise ValueError("internal prox dual has the wrong dimension")
        if np.any(~np.isfinite(dual)):
            raise ValueError("warm-start multiplier must be finite")
        self._warm_dual = dual.copy()
        self._has_warm_start = bool(self.rows or budget_eta is not None)
        self._warm_budget_eta = (
            None if budget_eta is None else float(budget_eta)
        )

    def export_warm_start(self) -> Dict[str, Any]:
        """Return a portable copy of the committed proximal state."""
        return {
            "internal_dual": self._warm_dual.copy(),
            "has_warm_start": bool(self._has_warm_start),
            "budget_eta": self._warm_budget_eta,
            "row_norms": self.row_norms.copy(),
        }

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

    def _lbfgs_coordinates(self, dual: np.ndarray) -> np.ndarray:
        """Map one support-function multiplier to smooth split variables."""
        pieces = []
        if self._lbfgs_equal.size:
            pieces.append(dual[self._lbfgs_equal])
        if self._lbfgs_upper.size:
            pieces.append(
                np.maximum(dual[self._lbfgs_upper], 0.0)
            )
        if self._lbfgs_lower.size:
            pieces.append(
                np.maximum(-dual[self._lbfgs_lower], 0.0)
            )
        if not pieces:
            return np.empty(0, dtype=float)
        return np.concatenate(pieces)

    def _lbfgs_dual(self, coordinates: np.ndarray) -> np.ndarray:
        """Map equality/upper/lower variables to one signed multiplier."""
        dual = np.zeros(self.rows, dtype=float)
        cursor = 0
        count = self._lbfgs_equal.size
        if count:
            dual[self._lbfgs_equal] = coordinates[cursor : cursor + count]
            cursor += count
        count = self._lbfgs_upper.size
        if count:
            dual[self._lbfgs_upper] += coordinates[cursor : cursor + count]
            cursor += count
        count = self._lbfgs_lower.size
        if count:
            dual[self._lbfgs_lower] -= coordinates[cursor : cursor + count]
        return dual

    def _solve_dual_lbfgs(
        self,
        values: np.ndarray,
        gamma: float,
        active_tolerance: float,
        active_limit: int,
    ) -> LinearProxResult:
        """Solve the smooth split dual with bound-constrained L-BFGS.

        A finite interval row is represented by two nonnegative
        multipliers, an equality row by one free multiplier, and a one-sided
        row by one nonnegative multiplier.  This removes the nonsmooth
        interval support function exactly; applying plain L-BFGS directly
        to that support function would not be valid.
        """
        warm_started = self._has_warm_start
        initial = self._lbfgs_coordinates(self._warm_dual)
        equality_count = self._lbfgs_equal.size
        upper_count = self._lbfgs_upper.size
        lower_count = self._lbfgs_lower.size
        bounds = (
            [(None, None)] * equality_count
            + [(0.0, None)] * (upper_count + lower_count)
        )
        evaluations = 0
        cached_coordinates: Optional[np.ndarray] = None
        cached_point: Optional[np.ndarray] = None
        cached_rows: Optional[np.ndarray] = None

        def objective_gradient(
            coordinates: np.ndarray,
        ) -> tuple[float, np.ndarray]:
            nonlocal evaluations
            nonlocal cached_coordinates, cached_point, cached_rows
            dual = self._lbfgs_dual(coordinates)
            shift = np.asarray(
                self.scaled_transpose @ dual,
                dtype=float,
            ).reshape(-1)
            shifted_argument = values - shift
            point = self._pava(shifted_argument, gamma)
            rows = np.asarray(
                self.scaled_operator @ point,
                dtype=float,
            ).reshape(-1)
            value = _smooth_dual_value(
                shifted_argument,
                point,
                gamma,
                self.k,
            )
            gradient_parts = []
            if equality_count:
                equality = self._lbfgs_equal
                value += float(
                    self.scaled_lower[equality]
                    @ coordinates[:equality_count]
                )
                gradient_parts.append(
                    self.scaled_lower[equality] - rows[equality]
                )
            cursor = equality_count
            if upper_count:
                upper = self._lbfgs_upper
                upper_coordinates = coordinates[
                    cursor : cursor + upper_count
                ]
                value += float(
                    self.scaled_upper[upper] @ upper_coordinates
                )
                gradient_parts.append(
                    self.scaled_upper[upper] - rows[upper]
                )
                cursor += upper_count
            if lower_count:
                lower = self._lbfgs_lower
                lower_coordinates = coordinates[
                    cursor : cursor + lower_count
                ]
                value -= float(
                    self.scaled_lower[lower] @ lower_coordinates
                )
                gradient_parts.append(
                    rows[lower] - self.scaled_lower[lower]
                )
            gradient = (
                np.concatenate(gradient_parts)
                if gradient_parts
                else np.empty(0, dtype=float)
            )
            evaluations += 1
            cached_coordinates = coordinates.copy()
            cached_point = point
            cached_rows = rows
            return float(value), gradient

        if initial.size:
            (
                final_coordinates,
                _,
                optimizer_information,
            ) = fmin_l_bfgs_b(
                objective_gradient,
                initial,
                bounds=bounds,
                m=self.lbfgs_memory,
                factr=0.0,
                pgtol=active_tolerance,
                maxfun=max(
                    20 * active_limit,
                    active_limit + 100,
                ),
                maxiter=active_limit,
                maxls=self.lbfgs_max_line_search,
            )
            final_coordinates = np.asarray(
                final_coordinates,
                dtype=float,
            ).reshape(-1)
            optimizer_status = str(
                optimizer_information.get("task", "")
            )
            completed = int(optimizer_information.get("nit", 0))
        else:
            final_coordinates = initial
            optimizer_status = "no bounded dual coordinates"
            completed = 0

        if (
            cached_coordinates is None
            or cached_coordinates.shape != final_coordinates.shape
            or not np.array_equal(
                cached_coordinates,
                final_coordinates,
            )
        ):
            objective_gradient(final_coordinates)
        if cached_point is None or cached_rows is None:
            raise RuntimeError("L-BFGS dual evaluation returned no point")

        final_dual = self._lbfgs_dual(final_coordinates)
        local_lipschitz = max(
            self.lipschitz / (1.0 + gamma),
            self.lipschitz * 1e-12,
            1e-15,
        )
        step = 1.0 / local_lipschitz
        fixed_point = _support_prox(
            final_dual + step * cached_rows,
            step,
            self.scaled_lower,
            self.scaled_upper,
        )
        fixed_point_residual = float(
            np.linalg.norm(
                fixed_point - final_dual,
                ord=np.inf,
            )
            / step
        )
        scaled_constraint_violation = _interval_violation(
            cached_rows,
            self.scaled_lower,
            self.scaled_upper,
        )
        original_rows = cached_rows * self.row_norms
        constraint_violation = _interval_violation(
            original_rows,
            self.lower,
            self.upper,
        )
        converged = bool(
            fixed_point_residual <= active_tolerance
            and scaled_constraint_violation <= active_tolerance
        )
        self._warm_dual = final_dual.copy()
        self._has_warm_start = True
        return LinearProxResult(
            x=cached_point,
            constraint_multiplier=final_dual / self.row_norms,
            converged=converged,
            iterations=completed,
            pava_calls=evaluations,
            fixed_point_residual=fixed_point_residual,
            constraint_violation=constraint_violation,
            scaled_constraint_violation=scaled_constraint_violation,
            method=self.method,
            lipschitz=local_lipschitz,
            restarts=0,
            line_search_backtracks=0,
            warm_started=warm_started,
            function_evaluations=evaluations,
            fallback_used=False,
            optimizer_status=optimizer_status,
        )

    def _lbfgs_fallback_result(
        self,
        lbfgs_result: LinearProxResult,
        values: np.ndarray,
        gamma: float,
        active_tolerance: float,
        active_limit: int,
    ) -> LinearProxResult:
        """Safeguard an unfinished L-BFGS solve with dual FISTA."""
        fallback = LinearConstraintProx(
            self.matrix,
            self.lower,
            self.upper,
            self.k,
            pava_method=self.pava_method,
            tolerance=active_tolerance,
            max_iterations=active_limit,
            adaptive_restart=self.adaptive_restart,
            use_budget_fast_path=False,
            dual_solver="fista",
        )
        fallback._warm_dual = self._warm_dual.copy()
        fallback._has_warm_start = True
        result = fallback.solve(
            values,
            gamma,
            tolerance=active_tolerance,
            max_iterations=active_limit,
        )
        self._warm_dual = fallback._warm_dual.copy()
        self._has_warm_start = fallback._has_warm_start
        return LinearProxResult(
            x=result.x,
            constraint_multiplier=result.constraint_multiplier,
            converged=result.converged,
            iterations=(
                lbfgs_result.iterations + result.iterations
            ),
            pava_calls=lbfgs_result.pava_calls + result.pava_calls,
            fixed_point_residual=result.fixed_point_residual,
            constraint_violation=result.constraint_violation,
            scaled_constraint_violation=(
                result.scaled_constraint_violation
            ),
            method="row_scaled_dual_lbfgsb_then_fista",
            lipschitz=result.lipschitz,
            restarts=result.restarts,
            line_search_backtracks=result.line_search_backtracks,
            warm_started=lbfgs_result.warm_started,
            function_evaluations=(
                lbfgs_result.function_evaluations
                + result.function_evaluations
            ),
            fallback_used=True,
            optimizer_status=lbfgs_result.optimizer_status,
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

        if self.dual_solver == "lbfgs":
            lbfgs_result = self._solve_dual_lbfgs(
                values,
                gamma,
                active_tolerance,
                active_limit,
            )
            if lbfgs_result.converged or not self.lbfgs_fallback:
                return lbfgs_result
            return self._lbfgs_fallback_result(
                lbfgs_result,
                values,
                gamma,
                active_tolerance,
                active_limit,
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
    dual_solver: str = "lbfgs",
    lbfgs_memory: int = 10,
    lbfgs_max_line_search: int = 40,
    lbfgs_fallback: bool = True,
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
        dual_solver=dual_solver,
        lbfgs_memory=lbfgs_memory,
        lbfgs_max_line_search=lbfgs_max_line_search,
        lbfgs_fallback=lbfgs_fallback,
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
    dual_solver: str = "lbfgs",
    lbfgs_memory: int = 10,
    lbfgs_max_line_search: int = 40,
    lbfgs_fallback: bool = True,
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
        dual_solver=dual_solver,
        lbfgs_memory=lbfgs_memory,
        lbfgs_max_line_search=lbfgs_max_line_search,
        lbfgs_fallback=lbfgs_fallback,
    ).x


__all__ = [
    "LinearConstraintProx",
    "LinearProxResult",
    "prox_linear",
    "prox_linear_details",
]
