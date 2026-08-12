"""Independent feasibility and objective checks for sparse incumbents."""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import numpy as np

from ..relaxation.problem import MarkowitzInstance


def _selector_vector(
    selectors: Optional[Any],
    x: np.ndarray,
    active_tolerance: float,
) -> np.ndarray:
    if selectors is None:
        return (np.abs(x) > active_tolerance).astype(float)
    array = np.asarray(selectors)
    if array.shape == x.shape:
        return np.asarray(array, dtype=float).reshape(-1)
    indices = np.asarray(selectors, dtype=np.int64).reshape(-1)
    if np.any(indices < 0) or np.any(indices >= x.size):
        raise ValueError("selector support contains an invalid index")
    result = np.zeros(x.size, dtype=float)
    result[np.unique(indices)] = 1.0
    return result


def sparse_objective(instance: MarkowitzInstance, x: Any) -> float:
    """Evaluate the original binary-perspective Markowitz objective."""
    vector = np.asarray(x, dtype=float).reshape(-1)
    if vector.shape != (instance.dimension,):
        raise ValueError("solution has the wrong dimension")
    exposure = np.asarray(instance.B.T @ vector, dtype=float).reshape(-1)
    return (
        0.5 * float(exposure @ exposure)
        + 0.5
        * float(instance.perspective_weight)
        * float(vector @ vector)
        - float(instance.return_reward) * float(instance.mu @ vector)
    )


def evaluate_incumbent(
    instance: MarkowitzInstance,
    x: Any,
    selectors: Optional[Any] = None,
    *,
    feasibility_tolerance: float = 1e-7,
    active_tolerance: float = 1e-9,
    required_assets: Optional[Sequence[int]] = None,
    forbidden_assets: Optional[Sequence[int]] = None,
) -> dict[str, Any]:
    """Recompute cardinality, original rows, and the candidate upper bound.

    ``numerically_feasible`` is intentionally named: a floating-point QP
    solution is verified against the requested tolerance, but this routine
    does not perform interval-arithmetic certification.
    """
    vector = np.asarray(x, dtype=float).reshape(-1)
    if vector.shape != (instance.dimension,):
        raise ValueError("solution has the wrong dimension")
    finite = bool(np.all(np.isfinite(vector)))
    if not finite:
        return {
            "objective": None,
            "upper_bound": None,
            "numerically_feasible": False,
            "active_support": np.empty(0, dtype=np.int64),
            "selectors": None,
            "violations": {"nonfinite": math.inf, "maximum": math.inf},
        }

    selector = _selector_vector(selectors, vector, active_tolerance)
    binary_violation = float(
        np.max(np.minimum(np.abs(selector), np.abs(selector - 1.0)))
    )
    selector_clipped = np.clip(np.rint(selector), 0.0, 1.0)
    active = np.flatnonzero(np.abs(vector) > active_tolerance)
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
    linear_violation = (
        max(float(np.max(lower_error)), float(np.max(upper_error)))
        if values.size
        else 0.0
    )
    required = np.asarray(
        tuple(() if required_assets is None else required_assets),
        dtype=np.int64,
    ).reshape(-1)
    forbidden = np.asarray(
        tuple(() if forbidden_assets is None else forbidden_assets),
        dtype=np.int64,
    ).reshape(-1)
    required_violation = (
        float(np.max(1.0 - selector_clipped[required]))
        if required.size
        else 0.0
    )
    forbidden_violation = (
        float(np.max(selector_clipped[forbidden]))
        if forbidden.size
        else 0.0
    )
    violations = {
        "linear_rows": linear_violation,
        "nonnegativity": max(0.0, -float(np.min(vector))),
        "upper_box": max(0.0, float(np.max(vector)) - 1.0),
        "active_cardinality": max(0.0, float(active.size - instance.k)),
        "selector_cardinality": max(
            0.0,
            float(np.sum(selector_clipped) - instance.k),
        ),
        "selector_binary": binary_violation,
        "selector_link": float(
            np.max(np.maximum(vector - selector_clipped, 0.0))
        ),
        "required_assets": required_violation,
        "forbidden_assets": forbidden_violation,
    }
    violations["maximum"] = max(violations.values())
    objective = sparse_objective(instance, vector)
    feasible = bool(
        math.isfinite(objective)
        and violations["maximum"] <= float(feasibility_tolerance)
    )
    return {
        "objective": objective,
        "upper_bound": objective if feasible else None,
        "numerically_feasible": feasible,
        "active_support": active,
        "selector_support": np.flatnonzero(selector_clipped > 0.5),
        "selectors": selector_clipped,
        "cardinality": int(active.size),
        "selector_cardinality": int(np.sum(selector_clipped)),
        "risk": 0.5
        * float(
            np.asarray(instance.B.T @ vector).reshape(-1)
            @ np.asarray(instance.B.T @ vector).reshape(-1)
        ),
        "ridge": 0.5
        * float(instance.perspective_weight)
        * float(vector @ vector),
        "expected_return": float(instance.mu @ vector),
        "constraint_values": values,
        "violations": violations,
        "feasibility_tolerance": float(feasibility_tolerance),
        "active_tolerance": float(active_tolerance),
        "floating_point_certified": False,
    }


__all__ = ["evaluate_incumbent", "sparse_objective"]
