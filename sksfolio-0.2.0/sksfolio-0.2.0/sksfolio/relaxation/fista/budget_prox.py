"""Perspective proximal oracle with an exact full-investment constraint.

For the package's long-only perspective function ``G_k``, this module solves

    minimize_x  0.5 * ||x - v||^2 + gamma * G_k(x)
    subject to  1.T @ x = 1.

If ``eta`` multiplies ``1.T @ x - 1``, the unique primal solution is

    x(eta) = prox_{gamma G_k}(v - eta * 1).

The scalar function ``sum(x(eta)) - 1`` is continuous and nonincreasing.
Brent's bracketed method is used by default; deterministic bisection remains
available and is also used as a safeguarded fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy.optimize import brentq

from ..pdhg.pava._common import (
    singleton_subtraction,
    validate as _validate_pava,
)
from ..pdhg.pava.full_sort import _pool
from ..pdhg.pava.partial_sort import (
    _mixed_pool_boundary,
    native_enabled as _native_partial_enabled,
    prox_native as _native_partial_prox,
)


@dataclass(frozen=True)
class BudgetProxResult:
    """A budget proximal point and its scalar equality certificate."""

    x: np.ndarray
    eta: float
    evaluations: int
    method: str
    root_method: str
    equality_residual: float
    warm_started: bool = False


def _normalize_method(method: str) -> str:
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


def _normalize_root_method(method: str) -> str:
    normalized = str(method).lower().replace("-", "_")
    normalized = {"brentq": "brent"}.get(normalized, normalized)
    if normalized not in {"brent", "bisection"}:
        raise ValueError(
            "root_method must be 'brent' or 'bisection'"
        )
    return normalized


def _simplex_projection(
    values: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Project onto the unit simplex and return its threshold."""
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered, dtype=np.float64)
    indices = np.arange(1, values.size + 1, dtype=np.float64)
    candidates = ordered - (cumulative - 1.0) / indices
    active = np.flatnonzero(candidates > 0.0)
    if active.size == 0:
        raise ArithmeticError("failed to locate the simplex threshold")
    count = int(active[-1]) + 1
    threshold = (
        math.fsum(float(value) for value in ordered[:count]) - 1.0
    ) / float(count)
    point = np.maximum(values - threshold, 0.0)
    return point, float(threshold)


class _CachedPAVA:
    """Evaluate shifted PAVA points while reusing shift-invariant ordering."""

    def __init__(
        self,
        values: np.ndarray,
        gamma: float,
        k: int,
        method: str,
    ) -> None:
        self.values = values
        self.gamma = gamma
        self.k = k
        self.method = method
        self.dimension = values.size
        self.native_partial = bool(
            method == "partial_sort" and _native_partial_enabled()
        )
        if method == "full_sort":
            self.order = np.argsort(-values, kind="stable")
            self.top_indices = np.empty(0, dtype=int)
            self.top_mask = np.empty(0, dtype=bool)
        elif self.native_partial:
            self.order = np.empty(0, dtype=int)
            self.top_indices = np.empty(0, dtype=int)
            self.top_mask = np.empty(0, dtype=bool)
        else:
            self.order = np.empty(0, dtype=int)
            self.top_indices = np.argpartition(
                values,
                self.dimension - k,
            )[self.dimension - k :]
            self.top_mask = np.zeros(self.dimension, dtype=bool)
            self.top_mask[self.top_indices] = True

    def __call__(self, eta: float) -> np.ndarray:
        if self.native_partial:
            return _native_partial_prox(
                self.values - eta,
                self.gamma,
                self.k,
            )
        positive = np.maximum(self.values - eta, 0.0)
        if not np.any(positive):
            return np.zeros_like(positive)
        if self.method == "full_sort":
            return self._full_sort(positive)
        return self._partial_sort(positive)

    def _full_sort(self, positive: np.ndarray) -> np.ndarray:
        ordered = positive[self.order]
        pools = []
        for position, magnitude in enumerate(ordered):
            pools.append(
                _pool(
                    position,
                    position + 1,
                    int(position < self.k),
                    float(magnitude),
                    self.gamma,
                )
            )
            while (
                len(pools) >= 2
                and pools[-2].subtraction < pools[-1].subtraction
            ):
                right = pools.pop()
                left = pools.pop()
                pools.append(
                    _pool(
                        left.start,
                        right.stop,
                        left.selected + right.selected,
                        left.total + right.total,
                        self.gamma,
                    )
                )

        subtraction = np.empty_like(ordered)
        for pool in pools:
            subtraction[pool.start : pool.stop] = pool.subtraction
        ordered_result = ordered - subtraction
        result = np.empty_like(positive)
        result[self.order] = ordered_result
        np.maximum(result, 0.0, out=result)
        np.minimum(result, 1.0, out=result)
        return result

    def _partial_sort(self, positive: np.ndarray) -> np.ndarray:
        top = positive[self.top_indices]
        tail = positive[~self.top_mask]
        boundary = _mixed_pool_boundary(top, tail, self.gamma)
        result = np.maximum(positive - boundary, 0.0)
        result[self.top_indices] = top - np.maximum(
            singleton_subtraction(top, self.gamma),
            boundary,
        )
        np.maximum(result, 0.0, out=result)
        np.minimum(result, 1.0, out=result)
        return result


class _RootOracle:
    """Count shifted PAVA calls and retain the best scalar evaluation."""

    def __init__(self, pava: _CachedPAVA) -> None:
        self.pava = pava
        self.evaluations = 0
        self.residual_cache: dict[float, float] = {}
        self.last_eta: float | None = None
        self.last_x: np.ndarray | None = None
        self.best_eta: float | None = None
        self.best_residual = math.inf

    def residual(self, eta: float) -> float:
        eta = float(eta)
        cached = self.residual_cache.get(eta)
        if cached is not None:
            return cached
        point = self.pava(eta)
        residual = math.fsum(float(value) for value in point) - 1.0
        self.evaluations += 1
        self.residual_cache[eta] = residual
        self.last_eta = eta
        self.last_x = point
        if abs(residual) < abs(self.best_residual):
            self.best_eta = eta
            self.best_residual = residual
        return residual

    def point(self, eta: float) -> np.ndarray:
        eta = float(eta)
        if self.last_eta == eta and self.last_x is not None:
            return self.last_x.copy()
        point = self.pava(eta)
        residual = math.fsum(float(value) for value in point) - 1.0
        self.evaluations += 1
        self.residual_cache[eta] = residual
        self.last_eta = eta
        self.last_x = point
        if abs(residual) < abs(self.best_residual):
            self.best_eta = eta
            self.best_residual = residual
        return point.copy()


def _bisection(
    oracle: _RootOracle,
    lower: float,
    upper: float,
    tolerance: float,
    max_iterations: int,
) -> float:
    """Safeguarded monotone bisection with a feasibility stopping rule."""
    lower_value = oracle.residual(lower)
    upper_value = oracle.residual(upper)
    if lower_value < 0.0 or upper_value > 0.0:
        raise ArithmeticError("failed to bracket the budget multiplier")
    if abs(lower_value) <= tolerance:
        return lower
    if abs(upper_value) <= tolerance:
        return upper

    root_tolerance = tolerance / float(oracle.pava.dimension)
    for _ in range(max_iterations):
        middle = lower + 0.5 * (upper - lower)
        middle_value = oracle.residual(middle)
        if abs(middle_value) <= tolerance:
            return middle
        if middle_value > 0.0:
            lower = middle
        else:
            upper = middle
        if upper - lower <= root_tolerance:
            return lower + 0.5 * (upper - lower)
    raise RuntimeError(
        "budget multiplier bisection exceeded max_iterations"
    )


def prox_budget_details(
    argument: Any,
    gamma: float,
    k: int,
    method: str = "partial_sort",
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
    root_method: str = "brent",
    initial_eta: float | None = None,
) -> BudgetProxResult:
    """Compute the budget-constrained prox and scalar multiplier details."""
    values = np.asarray(argument, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("argument must be nonempty")
    if np.any(~np.isfinite(values)):
        raise ValueError("argument must be finite")
    gamma = float(gamma)
    if not math.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("gamma must be positive and finite")
    if isinstance(k, (bool, np.bool_)) or int(k) != k:
        raise ValueError("k must be an integer")
    k = int(k)
    if not 1 <= k <= values.size:
        raise ValueError("k must lie in {1, ..., dimension}")
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive and finite")
    max_iterations = int(max_iterations)
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    method = _normalize_method(method)
    root_method = _normalize_root_method(root_method)
    if initial_eta is not None:
        initial_eta = float(initial_eta)
        if not math.isfinite(initial_eta):
            raise ValueError("initial_eta must be finite")

    minimum = float(np.min(values))
    maximum = float(np.max(values))
    spread = maximum - minimum
    if not math.isfinite(spread):
        raise ValueError(
            "argument range overflows Float64; rescale the data"
        )
    center = minimum + 0.5 * spread
    centered = values - center
    lower = float(np.min(centered) - (1.0 + gamma))
    upper = float(np.max(centered))
    _validate_pava(centered - lower, gamma, k)

    if k == 1:
        point, threshold = _simplex_projection(centered)
        eta = center + threshold - gamma
        return BudgetProxResult(
            x=point,
            eta=float(eta),
            evaluations=0,
            method=method,
            root_method=root_method,
            equality_residual=abs(
                math.fsum(float(value) for value in point) - 1.0
            ),
        )

    if k == values.size:
        point, threshold = _simplex_projection(
            centered / (1.0 + gamma)
        )
        eta = center + (1.0 + gamma) * threshold
        return BudgetProxResult(
            x=point,
            eta=float(eta),
            evaluations=0,
            method=method,
            root_method=root_method,
            equality_residual=abs(
                math.fsum(float(value) for value in point) - 1.0
            ),
        )

    pava = _CachedPAVA(centered, gamma, k, method)
    oracle = _RootOracle(pava)
    bracket_lower = lower
    bracket_upper = upper
    warm_started = initial_eta is not None
    warm_root = (
        None
        if initial_eta is None
        else min(max(initial_eta - center, lower), upper)
    )
    root: float
    warm_value: float | None = None
    if warm_root is not None:
        warm_value = oracle.residual(warm_root)
        if abs(warm_value) <= tolerance:
            root = warm_root
            bracket_lower = warm_root
            bracket_upper = warm_root
        else:
            radius = max(
                (upper - lower) / 64.0,
                tolerance / float(values.size),
            )
            if warm_value > 0.0:
                bracket_lower = warm_root
                bracket_upper = min(upper, warm_root + radius)
                while (
                    bracket_upper < upper
                    and oracle.residual(bracket_upper) > 0.0
                ):
                    radius *= 2.0
                    bracket_upper = min(upper, warm_root + radius)
            else:
                bracket_upper = warm_root
                bracket_lower = max(lower, warm_root - radius)
                while (
                    bracket_lower > lower
                    and oracle.residual(bracket_lower) < 0.0
                ):
                    radius *= 2.0
                    bracket_lower = max(lower, warm_root - radius)
    lower_value = oracle.residual(bracket_lower)
    if abs(lower_value) <= tolerance:
        root = bracket_lower
    else:
        upper_value = oracle.residual(bracket_upper)
        if lower_value < 0.0 or upper_value > 0.0:
            raise ArithmeticError(
                "failed to bracket the budget multiplier"
            )
        if root_method == "bisection":
            root = _bisection(
                oracle,
                bracket_lower,
                bracket_upper,
                tolerance,
                max_iterations,
            )
        else:
            root_tolerance = max(
                tolerance / float(values.size),
                np.finfo(np.float64).tiny,
            )
            root, report = brentq(
                oracle.residual,
                bracket_lower,
                bracket_upper,
                xtol=root_tolerance,
                rtol=4.0 * np.finfo(np.float64).eps,
                maxiter=max_iterations,
                full_output=True,
                disp=False,
            )
            root = float(root)
            if (
                not report.converged
                or abs(oracle.residual(root)) > tolerance
            ):
                root = _bisection(
                    oracle,
                    bracket_lower,
                    bracket_upper,
                    tolerance,
                    max_iterations,
                )

    point = oracle.point(root)
    residual = abs(
        math.fsum(float(value) for value in point) - 1.0
    )
    if residual > tolerance:
        if (
            oracle.best_eta is not None
            and abs(oracle.best_residual) < residual
        ):
            root = oracle.best_eta
            point = oracle.point(root)
            residual = abs(
                math.fsum(float(value) for value in point) - 1.0
            )
    if residual > tolerance:
        raise RuntimeError(
            "budget proximal oracle did not reach the requested tolerance"
        )
    return BudgetProxResult(
        x=point,
        eta=float(center + root),
        evaluations=oracle.evaluations,
        method=method,
        root_method=root_method,
        equality_residual=float(residual),
        warm_started=warm_started,
    )


def prox_budget(
    argument: Any,
    gamma: float,
    k: int,
    method: str = "partial_sort",
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
    root_method: str = "brent",
    initial_eta: float | None = None,
) -> np.ndarray:
    """Return ``prox_{gamma G_k + delta_{1.T x = 1}}(argument)``."""
    return prox_budget_details(
        argument,
        gamma,
        k,
        method,
        tolerance=tolerance,
        max_iterations=max_iterations,
        root_method=root_method,
        initial_eta=initial_eta,
    ).x


__all__ = [
    "BudgetProxResult",
    "prox_budget",
    "prox_budget_details",
]
