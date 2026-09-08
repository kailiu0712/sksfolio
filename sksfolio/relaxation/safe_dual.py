"""Safe Fenchel dual bounds for the perspective relaxation."""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np

from .certificate import SafeDualCertificate


def _factor_adjoint(
    problem: Any,
    value: np.ndarray,
) -> np.ndarray:
    """Apply the factor adjoint in the original problem coordinates."""
    return np.asarray(problem.B @ value, dtype=float).reshape(-1)


def _safe_interval_dual(
    value: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Project tiny support-domain errors to a valid interval multiplier."""
    result = np.asarray(value, dtype=float).reshape(-1).copy()
    if result.size == 0:
        return result, 0.0
    if np.any(~np.isfinite(result)):
        raise ValueError("constraint dual contains a nonfinite value")
    original = result.copy()
    finite_lower = np.isfinite(lower)
    finite_upper = np.isfinite(upper)
    lower_only = finite_lower & ~finite_upper
    upper_only = ~finite_lower & finite_upper
    unbounded = ~finite_lower & ~finite_upper
    result[lower_only] = np.minimum(result[lower_only], 0.0)
    result[upper_only] = np.maximum(result[upper_only], 0.0)
    result[unbounded] = 0.0
    return result, float(np.max(np.abs(result - original)))


def _interval_support(
    value: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Evaluate the support without undefined zero-times-infinity terms."""
    if value.size == 0:
        return 0.0
    positive = value > 0.0
    negative = value < 0.0
    if np.any(positive & ~np.isfinite(upper)):
        return math.inf
    if np.any(negative & ~np.isfinite(lower)):
        return math.inf
    support = 0.0
    if np.any(positive):
        support += float(value[positive] @ upper[positive])
    if np.any(negative):
        support += float(value[negative] @ lower[negative])
    return support


def _scaled_perspective_conjugate(
    argument: np.ndarray,
    weight: float,
    k: int,
) -> float:
    """Evaluate ``(weight * G_k)^*`` by partial top-k selection."""
    if weight <= 0.0 or not math.isfinite(weight):
        raise ValueError("perspective weight must be positive and finite")
    values = np.asarray(argument, dtype=float).reshape(-1)
    if np.any(~np.isfinite(values)):
        return math.inf
    penalties = np.zeros_like(values)
    quadratic = (values > 0.0) & (values < weight)
    linear = values >= weight
    penalties[quadratic] = (
        0.5
        * values[quadratic]
        * (values[quadratic] / weight)
    )
    penalties[linear] = values[linear] - 0.5 * weight
    if k >= penalties.size:
        return float(np.sum(penalties))
    split = penalties.size - k
    return float(np.sum(np.partition(penalties, split)[split:]))


def _dual_bound(
    problem: Any,
    factor_dual: np.ndarray,
    constraint_dual: np.ndarray,
) -> Tuple[float, np.ndarray, float]:
    """Return a weak-duality bound and support-feasible multiplier."""
    factor = np.asarray(factor_dual, dtype=float).reshape(-1)
    if factor.shape != (problem.rank,):
        raise ValueError("factor dual has the wrong dimension")
    if np.any(~np.isfinite(factor)):
        return -math.inf, np.zeros(problem.rows), math.inf
    constraint, correction = _safe_interval_dual(
        constraint_dual,
        problem.lower,
        problem.upper,
    )
    adjoint = _factor_adjoint(problem, factor)
    if problem.rows:
        adjoint = adjoint + problem.C.T @ constraint
    conjugate_argument = (
        problem.return_reward * problem.mu
        - np.asarray(adjoint).reshape(-1)
    )
    conjugate = _scaled_perspective_conjugate(
        conjugate_argument,
        problem.perspective_weight,
        problem.k,
    )
    support = _interval_support(
        constraint,
        problem.lower,
        problem.upper,
    )
    value = (
        -0.5 * float(factor @ factor)
        - support
        - conjugate
    )
    return float(value), constraint, float(correction)


class SafeDualEvaluator:
    """Reusable original-scale Fenchel dual-bound evaluator.

    Preparing the row-scaled problem is appreciable work.  Solvers that form
    a certificate more than once should keep one evaluator instead of calling
    :func:`evaluate_dual_bound` repeatedly.
    """

    def __init__(self, instance: Any) -> None:
        instance.validate()
        self._problem = instance
        matrix = instance.C
        if hasattr(matrix, "multiply"):
            squared = np.asarray(
                matrix.multiply(matrix).sum(axis=1),
                dtype=float,
            ).reshape(-1)
        else:
            squared = np.sum(np.asarray(matrix, dtype=float) ** 2, axis=1)
        self._row_norms = np.sqrt(np.maximum(squared, 0.0))
        self._row_norms[self._row_norms == 0.0] = 1.0

    def evaluate(
        self,
        factor_dual: Any,
        constraint_dual: Any,
        *,
        constraint_dual_scaled: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate one safe bound without rebuilding the problem data."""
        problem = self._problem
        factor = np.asarray(factor_dual, dtype=float).reshape(-1)
        constraint = np.asarray(
            constraint_dual,
            dtype=float,
        ).reshape(-1)
        if constraint.shape != (problem.rows,):
            raise ValueError("constraint dual has the wrong dimension")
        if constraint_dual_scaled and self._row_norms.size:
            constraint = constraint / self._row_norms
        value, safe_original, correction = _dual_bound(
            problem,
            factor,
            constraint,
        )
        safe_scaled = safe_original * self._row_norms
        return {
            "dual_bound": (
                float(value) if math.isfinite(value) else None
            ),
            "factor_dual": factor.copy(),
            "constraint_dual_scaled": safe_scaled,
            "constraint_dual_original": safe_original,
            "dual_domain_correction": float(correction),
            "safe_in_exact_arithmetic": math.isfinite(value),
            "floating_point_certified": False,
        }


def evaluate_dual_bound(
    instance: Any,
    factor_dual: Any,
    constraint_dual: Any,
    *,
    constraint_dual_scaled: bool = False,
) -> Dict[str, Any]:
    """Recompute a safe Fenchel lower bound from supplied multipliers.

    By default ``constraint_dual`` is interpreted on the original, unscaled
    constraint rows. Set ``constraint_dual_scaled=True`` when supplying the
    public row-scaled multiplier ``dual_bound_constraint_scaled``. Both
    Both choices are evaluated in the original objective coordinates.
    """
    return SafeDualEvaluator(instance).evaluate(
        factor_dual,
        constraint_dual,
        constraint_dual_scaled=constraint_dual_scaled,
    )


__all__ = [
    "SafeDualCertificate",
    "SafeDualEvaluator",
    "evaluate_dual_bound",
]
