"""Warm-started OSQP oracle for the exact weak-majorization proximal QP.

For the long-only k-support perspective function, the proximal problem can
be written with a sorted k-vector ``w`` that weakly majorizes ``x``:

    minimize  0.5 * ||x - v||^2 + 0.5 * gamma * ||w||^2

    subject to
        x >= 0,
        1 >= w_1 >= ... >= w_k >= 0,
        T_j(x) <= sum_{r=1}^j w_r,  j = 1, ..., k - 1,
        sum(x) <= sum(w),
        lower <= C @ x <= upper.

Here ``T_j`` is the sum of the ``j`` largest coordinates.  The
Rockafellar--Uryasev representation

    T_j(x) = min_theta j * theta + sum_i (x_i - theta)_+

turns every top-j inequality into linear constraints using one ``theta_j``
and ``d`` nonnegative slacks.  The formulation uses inequalities throughout;
replacing weak majorization by equality would define a different function.

OSQP is optional and imported only when :class:`MajorizationQPOracle` is
constructed.  One oracle instance has fixed ``C``, interval bounds, and
``k``.  Calls to :meth:`solve` update only ``v`` and ``gamma``, preserving
the matrix sparsity pattern and the preceding solution as a warm start.
The class is stateful and is not thread-safe.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Mapping, Optional

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class MajorizationQPResult:
    """One proximal result and its OSQP diagnostics."""

    x: Optional[np.ndarray]
    w: Optional[np.ndarray]
    multiplier: Optional[np.ndarray]
    objective: Optional[float]
    has_solution: bool
    converged: bool
    status: str
    status_code: int
    iterations: int
    primal_residual: Optional[float]
    dual_residual: Optional[float]
    maximum_constraint_violation: Optional[float]
    linear_constraint_violation: Optional[float]
    majorization_violation: Optional[float]
    warm_start_used: bool
    setup_seconds: float
    update_seconds: float
    solve_seconds: float
    total_seconds: float
    rho_updates: int


@dataclass(frozen=True)
class MajorizationQPWarmStart:
    """Transactional snapshot of OSQP's unmodified primal-dual iterate."""

    primal: Optional[np.ndarray]
    dual: Optional[np.ndarray]
    has_previous: bool


def _finite_matrix(value: Any) -> sparse.csr_matrix:
    matrix = sparse.csr_matrix(value, dtype=np.float64)
    if len(matrix.shape) != 2:
        raise ValueError("constraint_matrix must be two-dimensional")
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    if np.any(~np.isfinite(matrix.data)):
        raise ValueError("constraint_matrix contains a nonfinite value")
    return matrix


def _vector(
    value: Any,
    length: int,
    name: str,
    *,
    allow_infinite: bool,
) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (length,):
        raise ValueError(f"{name} must have length {length}")
    if np.any(np.isnan(result)):
        raise ValueError(f"{name} contains NaN")
    if not allow_infinite and np.any(~np.isfinite(result)):
        raise ValueError(f"{name} contains a nonfinite value")
    return result


def _order_matrix(k: int) -> sparse.csr_matrix:
    if k <= 1:
        return sparse.csr_matrix((0, k), dtype=np.float64)
    rows = np.repeat(np.arange(k - 1), 2)
    columns = np.column_stack(
        (np.arange(k - 1), np.arange(1, k))
    ).reshape(-1)
    data = np.tile(np.array([1.0, -1.0]), k - 1)
    return sparse.csr_matrix(
        (data, (rows, columns)),
        shape=(k - 1, k),
    )


def _top_w_matrix(k: int) -> sparse.csr_matrix:
    rows = []
    columns = []
    for index in range(1, k):
        rows.extend([index - 1] * index)
        columns.extend(range(index))
    return sparse.csr_matrix(
        (
            -np.ones(len(rows), dtype=np.float64),
            (np.asarray(rows), np.asarray(columns)),
        ),
        shape=(k - 1, k),
    )


def _maximum_interval_violation(
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    if values.size == 0:
        return 0.0
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
    return max(float(np.max(lower_error)), float(np.max(upper_error)))


def _info_float(info: Any, *names: str) -> Optional[float]:
    for name in names:
        value = getattr(info, name, None)
        if value is not None:
            try:
                result = float(value)
            except (TypeError, ValueError):
                continue
            return result if math.isfinite(result) else None
    return None


class MajorizationQPOracle:
    """Reusable OSQP oracle for the exact long-only majorization lift."""

    method = "majorization_qp_osqp"
    exact_budget = False
    lipschitz = 0.0
    lipschitz_kind = "not_applicable"

    def __init__(
        self,
        constraint_matrix: Any,
        lower_bounds: Any,
        upper_bounds: Any,
        k: int,
        *,
        tolerance: float = 1e-8,
        max_iterations: int = 20_000,
        polish: bool = True,
        verbose: bool = False,
        max_lift_variables: int = 5_000_000,
        osqp_settings: Optional[Mapping[str, Any]] = None,
    ) -> None:
        try:
            import osqp
        except (ImportError, OSError) as error:
            raise ImportError(
                "MajorizationQPOracle requires the optional OSQP package; "
                "install it with `python -m pip install 'osqp>=1.0'`"
            ) from error

        setup_start = time.perf_counter()
        C = _finite_matrix(constraint_matrix)
        rows, dimension = map(int, C.shape)
        if dimension < 1:
            raise ValueError(
                "constraint_matrix must have at least one column"
            )
        if isinstance(k, (bool, np.bool_)) or int(k) != k:
            raise ValueError("k must be an integer")
        k = int(k)
        if not 1 <= k <= dimension:
            raise ValueError("k must lie in {1, ..., dimension}")
        lower = _vector(
            lower_bounds,
            rows,
            "lower_bounds",
            allow_infinite=True,
        )
        upper = _vector(
            upper_bounds,
            rows,
            "upper_bounds",
            allow_infinite=True,
        )
        if np.any(np.isposinf(lower)):
            raise ValueError("lower_bounds cannot contain positive infinity")
        if np.any(np.isneginf(upper)):
            raise ValueError("upper_bounds cannot contain negative infinity")
        if np.any(lower > upper):
            raise ValueError("a lower bound exceeds its upper bound")

        tolerance = float(tolerance)
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be positive and finite")
        max_iterations = int(max_iterations)
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if (
            isinstance(max_lift_variables, (bool, np.bool_))
            or int(max_lift_variables) != max_lift_variables
        ):
            raise ValueError("max_lift_variables must be an integer")
        max_lift_variables = int(max_lift_variables)
        if max_lift_variables < 1:
            raise ValueError("max_lift_variables must be positive")

        reserved = {
            "eps_abs",
            "eps_rel",
            "max_iter",
            "polishing",
            "warm_starting",
        }
        extra_settings = dict(osqp_settings or {})
        overlap = sorted(reserved.intersection(extra_settings))
        if overlap:
            raise ValueError(
                "set "
                + ", ".join(overlap)
                + " through the explicit oracle arguments"
            )

        theta_count = k - 1
        slack_count = theta_count * dimension
        x_offset = 0
        w_offset = dimension
        theta_offset = w_offset + k
        slack_offset = theta_offset + theta_count
        variable_count = slack_offset + slack_count
        if variable_count > max_lift_variables:
            raise ValueError(
                "the exact weak-majorization RU lift requires "
                f"{variable_count:,} variables for d={dimension:,} and "
                f"k={k:,}, exceeding max_lift_variables="
                f"{max_lift_variables:,}; this lift has O(d * k) memory, "
                "so use the PAVA-based proximal oracle at this scale"
            )

        zero = sparse.csr_matrix
        identity_x = sparse.eye(
            dimension,
            format="csr",
            dtype=np.float64,
        )
        identity_w = sparse.eye(k, format="csr", dtype=np.float64)
        identity_theta = sparse.eye(
            theta_count,
            format="csr",
            dtype=np.float64,
        )
        identity_slack = sparse.eye(
            slack_count,
            format="csr",
            dtype=np.float64,
        )

        matrices = []
        lower_chunks = []
        upper_chunks = []

        matrices.append(
            sparse.hstack(
                (
                    identity_x,
                    zero((dimension, k)),
                    zero((dimension, theta_count)),
                    zero((dimension, slack_count)),
                ),
                format="csr",
            )
        )
        lower_chunks.append(np.zeros(dimension))
        upper_chunks.append(np.full(dimension, np.inf))

        matrices.append(
            sparse.hstack(
                (
                    zero((k, dimension)),
                    identity_w,
                    zero((k, theta_count)),
                    zero((k, slack_count)),
                ),
                format="csr",
            )
        )
        lower_chunks.append(np.zeros(k))
        upper_chunks.append(np.ones(k))

        if theta_count:
            order = _order_matrix(k)
            matrices.append(
                sparse.hstack(
                    (
                        zero((theta_count, dimension)),
                        order,
                        zero((theta_count, theta_count)),
                        zero((theta_count, slack_count)),
                    ),
                    format="csr",
                )
            )
            lower_chunks.append(np.zeros(theta_count))
            upper_chunks.append(np.full(theta_count, np.inf))

            repeated_x = sparse.kron(
                sparse.csr_matrix(
                    np.ones((theta_count, 1), dtype=np.float64)
                ),
                -identity_x,
                format="csr",
            )
            repeated_theta = sparse.kron(
                identity_theta,
                sparse.csr_matrix(
                    np.ones((dimension, 1), dtype=np.float64)
                ),
                format="csr",
            )
            matrices.append(
                sparse.hstack(
                    (
                        repeated_x,
                        zero((slack_count, k)),
                        repeated_theta,
                        identity_slack,
                    ),
                    format="csr",
                )
            )
            lower_chunks.append(np.zeros(slack_count))
            upper_chunks.append(np.full(slack_count, np.inf))

            matrices.append(
                sparse.hstack(
                    (
                        zero((slack_count, dimension)),
                        zero((slack_count, k)),
                        zero((slack_count, theta_count)),
                        identity_slack,
                    ),
                    format="csr",
                )
            )
            lower_chunks.append(np.zeros(slack_count))
            upper_chunks.append(np.full(slack_count, np.inf))

            top_w = _top_w_matrix(k)
            top_theta = sparse.diags(
                np.arange(1, k, dtype=np.float64),
                format="csr",
            )
            top_slack = sparse.kron(
                identity_theta,
                sparse.csr_matrix(
                    np.ones((1, dimension), dtype=np.float64)
                ),
                format="csr",
            )
            matrices.append(
                sparse.hstack(
                    (
                        zero((theta_count, dimension)),
                        top_w,
                        top_theta,
                        top_slack,
                    ),
                    format="csr",
                )
            )
            lower_chunks.append(np.full(theta_count, -np.inf))
            upper_chunks.append(np.zeros(theta_count))

        matrices.append(
            sparse.csr_matrix(
                np.concatenate(
                    (
                        np.ones(dimension),
                        -np.ones(k),
                        np.zeros(theta_count + slack_count),
                    )
                ).reshape(1, -1)
            )
        )
        lower_chunks.append(np.array([-np.inf]))
        upper_chunks.append(np.array([0.0]))

        c_row_start = sum(matrix.shape[0] for matrix in matrices)
        if rows:
            matrices.append(
                sparse.hstack(
                    (
                        C,
                        zero((rows, k)),
                        zero((rows, theta_count)),
                        zero((rows, slack_count)),
                    ),
                    format="csr",
                )
            )
            lower_chunks.append(lower.copy())
            upper_chunks.append(upper.copy())
        c_row_slice = slice(c_row_start, c_row_start + rows)

        A = sparse.vstack(matrices, format="csc")
        constraint_lower = np.concatenate(lower_chunks)
        constraint_upper = np.concatenate(upper_chunks)
        diagonal_indices = np.arange(dimension + k)
        P = sparse.csc_matrix(
            (
                np.ones(dimension + k),
                (diagonal_indices, diagonal_indices),
            ),
            shape=(variable_count, variable_count),
        )
        initial_linear = np.zeros(variable_count)

        solver = osqp.OSQP()
        settings = {
            "verbose": bool(verbose),
            "eps_abs": tolerance,
            "eps_rel": tolerance,
            "max_iter": max_iterations,
            "polishing": bool(polish),
            "warm_starting": True,
        }
        settings.update(extra_settings)
        solver.setup(
            P=P,
            q=initial_linear,
            A=A,
            l=constraint_lower,
            u=constraint_upper,
            **settings,
        )

        self._solver = solver
        self._C = C
        self._lower = lower
        self._upper = upper
        self._A = A
        self._constraint_lower = constraint_lower
        self._constraint_upper = constraint_upper
        self._c_row_slice = c_row_slice
        self._dimension = dimension
        self._rows = rows
        self._k = k
        self._variable_count = variable_count
        self._constraint_count = int(A.shape[0])
        self._w_offset = w_offset
        self._tolerance = tolerance
        self._max_iterations = max_iterations
        self._has_previous = False
        self._last_raw_primal: Optional[np.ndarray] = None
        self._last_raw_dual: Optional[np.ndarray] = None
        self._setup_seconds = time.perf_counter() - setup_start

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def k(self) -> int:
        return self._k

    @property
    def setup_seconds(self) -> float:
        return self._setup_seconds

    def reset_warm_start(self) -> None:
        """Discard the preceding OSQP iterate."""
        self._solver.warm_start(
            x=np.zeros(self._variable_count),
            y=np.zeros(self._constraint_count),
        )
        self._has_previous = False
        self._last_raw_primal = None
        self._last_raw_dual = None

    def reset(self) -> None:
        """Alias for :meth:`reset_warm_start`."""
        self.reset_warm_start()

    def snapshot(self) -> MajorizationQPWarmStart:
        """Copy the current OSQP iterate for transactional restoration."""
        return MajorizationQPWarmStart(
            primal=(
                None
                if self._last_raw_primal is None
                else self._last_raw_primal.copy()
            ),
            dual=(
                None
                if self._last_raw_dual is None
                else self._last_raw_dual.copy()
            ),
            has_previous=bool(self._has_previous),
        )

    def restore(self, state: MajorizationQPWarmStart) -> None:
        """Restore a snapshot after a rejected outer line-search trial."""
        if not isinstance(state, MajorizationQPWarmStart):
            raise TypeError(
                "state must be a MajorizationQPWarmStart returned by "
                "snapshot()"
            )
        primal = (
            None
            if state.primal is None
            else np.asarray(state.primal, dtype=np.float64).reshape(-1)
        )
        dual = (
            None
            if state.dual is None
            else np.asarray(state.dual, dtype=np.float64).reshape(-1)
        )
        if primal is not None and (
            primal.shape != (self._variable_count,)
            or np.any(~np.isfinite(primal))
        ):
            raise ValueError("snapshot primal iterate is invalid")
        if dual is not None and (
            dual.shape != (self._constraint_count,)
            or np.any(~np.isfinite(dual))
        ):
            raise ValueError("snapshot dual iterate is invalid")
        if primal is None and dual is None:
            self.reset_warm_start()
            return
        warm_start: dict[str, np.ndarray] = {}
        if primal is not None:
            warm_start["x"] = primal
        if dual is not None:
            warm_start["y"] = dual
        self._solver.warm_start(**warm_start)
        self._last_raw_primal = (
            None if primal is None else primal.copy()
        )
        self._last_raw_dual = None if dual is None else dual.copy()
        self._has_previous = bool(state.has_previous)

    def solve(
        self,
        argument: Any,
        gamma: float,
        *,
        warm_start: bool = True,
        tolerance: Optional[float] = None,
        max_iterations: Optional[int] = None,
    ) -> MajorizationQPResult:
        """Solve one proximal QP while reusing the assembled OSQP model."""
        call_start = time.perf_counter()
        values = _vector(
            argument,
            self._dimension,
            "argument",
            allow_infinite=False,
        )
        gamma = float(gamma)
        if not math.isfinite(gamma) or gamma <= 0.0:
            raise ValueError("gamma must be positive and finite")
        effective_tolerance = (
            self._tolerance if tolerance is None else float(tolerance)
        )
        if (
            not math.isfinite(effective_tolerance)
            or effective_tolerance <= 0.0
        ):
            raise ValueError("tolerance must be positive and finite")
        effective_iterations = (
            self._max_iterations
            if max_iterations is None
            else int(max_iterations)
        )
        if effective_iterations < 1:
            raise ValueError("max_iterations must be positive")

        update_start = time.perf_counter()
        linear = np.zeros(self._variable_count)
        linear[: self._dimension] = -values
        quadratic = np.concatenate(
            (
                np.ones(self._dimension),
                np.full(self._k, gamma),
            )
        )
        self._solver.update(q=linear, Px=quadratic)
        self._solver.update_settings(
            eps_abs=effective_tolerance,
            eps_rel=effective_tolerance,
            max_iter=effective_iterations,
        )
        warm_start_used = bool(warm_start and self._has_previous)
        if not warm_start:
            self.reset_warm_start()
        update_seconds = time.perf_counter() - update_start

        solve_start = time.perf_counter()
        raw = self._solver.solve(raise_error=False)
        solve_seconds = time.perf_counter() - solve_start
        info = raw.info
        status_text = str(getattr(info, "status", "unknown"))
        status = status_text.lower().replace(" ", "_")
        status_code = int(getattr(info, "status_val", -1))
        primal_residual = _info_float(info, "prim_res", "pri_res")
        dual_residual = _info_float(info, "dual_res", "dua_res")
        converged = status_code == 1 or (
            status_code == 2
            and primal_residual is not None
            and dual_residual is not None
            and primal_residual <= 10.0 * effective_tolerance
            and dual_residual <= 10.0 * effective_tolerance
        )

        raw_point = getattr(raw, "x", None)
        raw_multiplier = getattr(raw, "y", None)
        point = (
            None
            if raw_point is None
            else np.asarray(raw_point, dtype=np.float64).reshape(-1)
        )
        dual = (
            None
            if raw_multiplier is None
            else np.asarray(
                raw_multiplier,
                dtype=np.float64,
            ).reshape(-1)
        )
        self._last_raw_primal = (
            None if point is None else point.copy()
        )
        self._last_raw_dual = None if dual is None else dual.copy()
        has_solution = bool(
            point is not None
            and point.shape == (self._variable_count,)
            and np.all(np.isfinite(point))
        )
        x: Optional[np.ndarray] = None
        w: Optional[np.ndarray] = None
        multiplier: Optional[np.ndarray] = None
        objective: Optional[float] = None
        maximum_violation: Optional[float] = None
        linear_violation: Optional[float] = None
        majorization_violation: Optional[float] = None
        if has_solution:
            assert point is not None
            x = point[: self._dimension].copy()
            w = point[
                self._w_offset : self._w_offset + self._k
            ].copy()
            objective = (
                0.5 * float((x - values) @ (x - values))
                + 0.5 * gamma * float(w @ w)
            )
            activities = np.asarray(self._A @ point).reshape(-1)
            maximum_violation = _maximum_interval_violation(
                activities,
                self._constraint_lower,
                self._constraint_upper,
            )
            linear_values = np.asarray(self._C @ x).reshape(-1)
            linear_violation = _maximum_interval_violation(
                linear_values,
                self._lower,
                self._upper,
            )
            ordered = np.sort(x)[::-1]
            prefix_x = np.cumsum(ordered)
            prefix_w = np.cumsum(w)
            top_error = (
                float(
                    np.max(
                        prefix_x[: self._k - 1]
                        - prefix_w[: self._k - 1]
                    )
                )
                if self._k > 1
                else 0.0
            )
            majorization_violation = max(
                0.0,
                -float(np.min(x)),
                float(np.max(w)) - 1.0,
                -float(np.min(w)),
                (
                    float(np.max(w[1:] - w[:-1]))
                    if self._k > 1
                    else 0.0
                ),
                top_error,
                float(np.sum(x) - np.sum(w)),
            )
            if (
                dual is not None
                and dual.shape == (self._constraint_count,)
                and np.all(np.isfinite(dual))
            ):
                multiplier = dual[self._c_row_slice].copy()

        self._has_previous = bool(converged and has_solution)
        iterations = int(getattr(info, "iter", 0))
        rho_updates = int(getattr(info, "rho_updates", 0))
        total_seconds = time.perf_counter() - call_start
        return MajorizationQPResult(
            x=x,
            w=w,
            multiplier=multiplier,
            objective=objective,
            has_solution=has_solution,
            converged=converged,
            status=status,
            status_code=status_code,
            iterations=iterations,
            primal_residual=primal_residual,
            dual_residual=dual_residual,
            maximum_constraint_violation=maximum_violation,
            linear_constraint_violation=linear_violation,
            majorization_violation=majorization_violation,
            warm_start_used=warm_start_used,
            setup_seconds=float(self._setup_seconds),
            update_seconds=float(update_seconds),
            solve_seconds=float(solve_seconds),
            total_seconds=float(total_seconds),
            rho_updates=rho_updates,
        )


__all__ = [
    "MajorizationQPOracle",
    "MajorizationQPResult",
    "MajorizationQPWarmStart",
]
