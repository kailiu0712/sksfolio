"""Shared helpers for native-Python commercial-solver implementations.

Both backends solve the same continuous long-only perspective relaxation

    minimize  0.5 ||B.T @ x||_2^2
              + 0.5 * perspective_weight * sum(t)
              - return_reward * mu.T @ x

    subject to
        x_i^2 <= t_i z_i,
        0 <= x_i <= z_i <= 1,
        sum(z) <= k,
        lower <= C x <= upper.

The factor representation is used directly; neither backend forms the dense
covariance matrix ``B @ B.T``.

Solver-specific model construction lives in ``gurobi.python`` and
``mosek.python``.  This module contains only their shared preparation,
diagnostic, result-normalization, and warm-start helpers.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from .state import RelaxationState


@dataclass(frozen=True)
class _PreparedInstance:
    B: Any
    mu: np.ndarray
    C: Any
    lower: np.ndarray
    upper: np.ndarray
    anchor: np.ndarray
    k: int
    perspective_weight: float
    return_reward: float
    dimension: int
    factors: int
    constraints: int
    constraint_names: Tuple[str, ...]


_BASE_OPTION_DEFAULTS: Dict[str, Any] = {
    "threads": 0,
    "tolerance": 1e-6,
    "time_limit": None,
    "log": False,
    "warm_start": True,
}


def _get_field(instance: Any, name: str) -> Any:
    if isinstance(instance, Mapping):
        if name not in instance:
            raise ValueError(f"instance is missing {name!r}")
        return instance[name]
    if not hasattr(instance, name):
        raise ValueError(f"instance is missing attribute {name!r}")
    return getattr(instance, name)


def _is_sparse_matrix(value: Any) -> bool:
    return (
        hasattr(value, "tocoo")
        and callable(value.tocoo)
        and hasattr(value, "shape")
    )


def _prepare_matrix(
    value: Any,
    name: str,
    expected_columns: Optional[int] = None,
) -> Any:
    if _is_sparse_matrix(value):
        matrix = value.astype(np.float64, copy=False)
        if len(matrix.shape) != 2:
            raise ValueError(f"{name} must be two-dimensional")
        if expected_columns is not None and matrix.shape[1] != expected_columns:
            raise ValueError(
                f"{name} has {matrix.shape[1]} columns; "
                f"expected {expected_columns}"
            )
        data = matrix.data if hasattr(matrix, "data") else matrix.tocoo().data
        if not np.all(np.isfinite(np.asarray(data, dtype=np.float64))):
            raise ValueError(f"{name} contains a non-finite coefficient")
        return matrix

    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if expected_columns is not None and matrix.shape[1] != expected_columns:
        raise ValueError(
            f"{name} has {matrix.shape[1]} columns; "
            f"expected {expected_columns}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains a non-finite coefficient")
    # Bundles memory-map B in Fortran order so that B.T is C-contiguous.
    # Preserve either native contiguous layout instead of duplicating a
    # potentially multi-gigabyte factor matrix.
    if matrix.flags.c_contiguous or matrix.flags.f_contiguous:
        return matrix
    return np.ascontiguousarray(matrix)


def _prepare_vector(
    value: Any,
    name: str,
    length: Optional[int] = None,
    allow_infinite: bool = False,
) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if length is not None and vector.size != length:
        raise ValueError(
            f"{name} has length {vector.size}; expected {length}"
        )
    if np.any(np.isnan(vector)):
        raise ValueError(f"{name} contains NaN")
    if not allow_infinite and not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains a non-finite value")
    return np.ascontiguousarray(vector)


def _prepare_instance(instance: Any) -> _PreparedInstance:
    B = _prepare_matrix(_get_field(instance, "B"), "B")
    dimension, factors = int(B.shape[0]), int(B.shape[1])
    if dimension < 1:
        raise ValueError("B must have at least one row")
    if factors < 1:
        raise ValueError("B must have at least one column")

    mu = _prepare_vector(
        _get_field(instance, "mu"),
        "mu",
        length=dimension,
    )
    C = _prepare_matrix(
        _get_field(instance, "C"),
        "C",
        expected_columns=dimension,
    )
    constraints = int(C.shape[0])
    lower = _prepare_vector(
        _get_field(instance, "lower"),
        "lower",
        length=constraints,
        allow_infinite=True,
    )
    upper = _prepare_vector(
        _get_field(instance, "upper"),
        "upper",
        length=constraints,
        allow_infinite=True,
    )
    if np.any(lower > upper):
        raise ValueError("a component of lower exceeds upper")
    if np.any(np.isposinf(lower)):
        raise ValueError("lower cannot contain positive infinity")
    if np.any(np.isneginf(upper)):
        raise ValueError("upper cannot contain negative infinity")

    anchor = _prepare_vector(
        _get_field(instance, "anchor"),
        "anchor",
        length=dimension,
    )
    raw_k = _get_field(instance, "k")
    try:
        k_float = float(raw_k)
    except (TypeError, ValueError) as error:
        raise ValueError("k must be an integer") from error
    if not math.isfinite(k_float) or not k_float.is_integer():
        raise ValueError("k must be an integer")
    k = int(k_float)
    if not 0 <= k <= dimension:
        raise ValueError("k must lie in {0, ..., d}")

    perspective_weight = float(
        _get_field(instance, "perspective_weight")
    )
    if (
        not math.isfinite(perspective_weight)
        or perspective_weight < 0.0
    ):
        raise ValueError("perspective_weight must be nonnegative")
    return_reward = float(_get_field(instance, "return_reward"))
    if not math.isfinite(return_reward) or return_reward < 0.0:
        raise ValueError("return_reward must be nonnegative")

    try:
        raw_constraint_names = _get_field(instance, "constraint_names")
    except ValueError:
        raw_constraint_names = tuple(
            f"row_{index}" for index in range(constraints)
        )
    constraint_names = tuple(str(name) for name in raw_constraint_names)
    if len(constraint_names) != constraints:
        raise ValueError(
            "constraint_names must contain one name per row"
        )
    if len(set(constraint_names)) != len(constraint_names):
        raise ValueError("constraint_names must be unique")

    return _PreparedInstance(
        B=B,
        mu=mu,
        C=C,
        lower=lower,
        upper=upper,
        anchor=anchor,
        k=k,
        perspective_weight=perspective_weight,
        return_reward=return_reward,
        dimension=dimension,
        factors=factors,
        constraints=constraints,
        constraint_names=constraint_names,
    )


def _options_mapping(options: Optional[Any]) -> Dict[str, Any]:
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


def _prepare_options(
    options: Optional[Any],
    solver_specific_key: str,
) -> Dict[str, Any]:
    supplied = _options_mapping(options)
    allowed = set(_BASE_OPTION_DEFAULTS) | {solver_specific_key}
    unknown = sorted(set(supplied) - allowed)
    if unknown:
        raise ValueError(
            "unknown solver option(s): " + ", ".join(unknown)
        )

    result = dict(_BASE_OPTION_DEFAULTS)
    result.update(
        {
            key: value
            for key, value in supplied.items()
            if key in _BASE_OPTION_DEFAULTS
        }
    )
    try:
        result["threads"] = int(result["threads"])
    except (TypeError, ValueError) as error:
        raise ValueError("threads must be an integer") from error
    if result["threads"] < 0:
        raise ValueError("threads must be nonnegative")

    result["tolerance"] = float(result["tolerance"])
    if (
        not math.isfinite(result["tolerance"])
        or not 0.0 < result["tolerance"] <= 1.0
    ):
        raise ValueError("tolerance must lie in (0, 1]")
    if result["time_limit"] is not None:
        result["time_limit"] = float(result["time_limit"])
        if (
            not math.isfinite(result["time_limit"])
            or result["time_limit"] <= 0.0
        ):
            raise ValueError("time_limit must be positive and finite")
    result["log"] = bool(result["log"])
    raw_warm_start = result["warm_start"]
    if isinstance(raw_warm_start, (bool, np.bool_)):
        result["warm_start"] = bool(raw_warm_start)
        result["warm_start_state"] = None
    elif raw_warm_start is None:
        result["warm_start"] = False
        result["warm_start_state"] = None
    else:
        result["warm_start"] = True
        result["warm_start_state"] = RelaxationState.coerce(
            raw_warm_start
        )

    raw_specific = supplied.get(solver_specific_key, {})
    if raw_specific is None:
        raw_specific = {}
    if not isinstance(raw_specific, Mapping):
        raise TypeError(f"{solver_specific_key} must be a mapping")
    result["solver_params"] = dict(raw_specific)
    return result


def _empty_result(
    solver: str,
    status: str,
    start: float,
    message: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "solver": solver,
        "status": status,
        "success": False,
        "has_solution": False,
        "x": None,
        "model_objective": None,
        "external_objective": None,
        "objective_scaling_gap": None,
        "violation": None,
        "build_seconds": None,
        "solve_seconds": None,
        "postprocess_seconds": None,
        "total_seconds": time.perf_counter() - start,
        "iterations": None,
        "message": message,
        "solver_details": {},
    }


def _failure_kind(error: BaseException) -> str:
    message = f"{type(error).__name__}: {error}".lower()
    license_markers = (
        "license",
        "flexlm",
        "no gurobi license",
        "err_missing_license",
        "err_license",
    )
    return (
        "license"
        if any(marker in message for marker in license_markers)
        else "solver"
    )


def _matrix_vector(matrix: Any, vector: np.ndarray) -> np.ndarray:
    return np.asarray(matrix @ vector, dtype=np.float64).reshape(-1)


def _factor_exposure(
    instance: _PreparedInstance,
    x: np.ndarray,
) -> np.ndarray:
    return np.asarray(instance.B.T @ x, dtype=np.float64).reshape(-1)


def _optimal_z(x: np.ndarray, k: int) -> Optional[np.ndarray]:
    """Return an optimal z for the implicit perspective value at feasible x."""
    x = np.asarray(x, dtype=np.float64)
    if (
        np.min(x, initial=0.0) < 0.0
        or np.max(x, initial=0.0) > 1.0
        or float(np.sum(x)) > float(k) + 1e-12
    ):
        return None
    z = np.zeros_like(x)
    positive = x > 0.0
    positive_count = int(np.count_nonzero(positive))
    if positive_count == 0:
        return z
    if positive_count <= k:
        z[positive] = 1.0
        return z

    values = x[positive]
    lower_scale = 1.0
    upper_scale = 2.0
    while float(np.minimum(1.0, upper_scale * values).sum()) < k:
        upper_scale *= 2.0
    for _ in range(64):
        scale = 0.5 * (lower_scale + upper_scale)
        if float(np.minimum(1.0, scale * values).sum()) < k:
            lower_scale = scale
        else:
            upper_scale = scale
    z[positive] = np.maximum(
        values,
        np.minimum(1.0, upper_scale * values),
    )
    return z


def _perspective_value(
    x: np.ndarray,
    k: int,
    tolerance: float,
) -> float:
    """Match ``instance_bundle.perspective_value`` without an import cycle."""
    vector = np.asarray(x, dtype=np.float64)
    if (
        float(np.min(vector, initial=0.0)) < -tolerance
        or float(np.max(vector, initial=0.0)) > 1.0 + tolerance
        or float(np.sum(vector)) > float(k) + tolerance
    ):
        return math.inf
    values = np.maximum(vector, 0.0)
    positive = values[values > 0.0]
    if positive.size == 0:
        return 0.0
    if positive.size <= k:
        return 0.5 * float(positive @ positive)

    lower_scale = 1.0
    upper_scale = max(
        2.0,
        float(k) / max(float(np.sum(positive)), 1e-16),
    )
    while float(np.minimum(1.0, upper_scale * positive).sum()) < k:
        upper_scale *= 2.0
    for _ in range(60):
        scale = 0.5 * (lower_scale + upper_scale)
        if float(np.minimum(1.0, scale * positive).sum()) < k:
            lower_scale = scale
        else:
            upper_scale = scale
    z = np.maximum(
        positive,
        np.minimum(1.0, upper_scale * positive),
    )
    if not np.all(z > 0.0):
        return math.inf
    return 0.5 * float(np.sum(positive * positive / z))


def _diagnostics(
    instance: _PreparedInstance,
    x: np.ndarray,
) -> Dict[str, Any]:
    if instance.constraints:
        cx = _matrix_vector(instance.C, x)
        finite_lower = np.isfinite(instance.lower)
        finite_upper = np.isfinite(instance.upper)
        lower_violation = (
            float(np.max(instance.lower[finite_lower] - cx[finite_lower]))
            if np.any(finite_lower)
            else 0.0
        )
        upper_violation = (
            float(np.max(cx[finite_upper] - instance.upper[finite_upper]))
            if np.any(finite_upper)
            else 0.0
        )
        linear_violation = max(0.0, lower_violation, upper_violation)
    else:
        linear_violation = 0.0
    nonnegativity = max(0.0, -float(np.min(x, initial=0.0)))
    upper_box = max(
        0.0,
        float(np.max(x, initial=0.0)) - 1.0,
    )
    perspective_budget = max(
        0.0,
        float(np.sum(x)) - float(instance.k),
    )
    violations = {
        "linear_rows": linear_violation,
        "nonnegativity": nonnegativity,
        "upper_box": upper_box,
        "perspective_budget": perspective_budget,
    }
    violations["maximum"] = max(violations.values())
    return {
        "linear_violation": linear_violation,
        "box_violation": max(nonnegativity, upper_box),
        "perspective_budget_violation": perspective_budget,
        "violation": violations["maximum"],
        "violations": violations,
    }


def _external_evaluation(
    instance: _PreparedInstance,
    x: np.ndarray,
    domain_tolerance: float,
) -> Dict[str, Any]:
    exposure = _factor_exposure(instance, x)
    risk = 0.5 * float(exposure @ exposure)
    perspective = _perspective_value(
        x,
        instance.k,
        tolerance=domain_tolerance,
    )
    expected_return = float(instance.mu @ x)
    objective = (
        risk
        + instance.perspective_weight * perspective
        - instance.return_reward * expected_return
    )
    diagnostics = _diagnostics(instance, x)
    return {
        "objective": objective if math.isfinite(objective) else None,
        "risk": risk,
        "expected_return": expected_return,
        "perspective_value": (
            perspective if math.isfinite(perspective) else None
        ),
        "budget": float(np.sum(x)),
        "maximum_weight": float(np.max(x, initial=0.0)),
        "positive_weights_above_1e-8": int(
            np.count_nonzero(x > 1e-8)
        ),
        "violations": diagnostics["violations"],
        "_diagnostics": diagnostics,
    }


def _warm_start_values(
    instance: _PreparedInstance,
    state: Optional[RelaxationState] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]]:
    x = (
        instance.anchor
        if state is None
        else state.compatible_primal(instance)
    )
    tolerance = 1e-10
    if (
        float(np.min(x, initial=0.0)) < -tolerance
        or float(np.max(x, initial=0.0)) > 1.0 + tolerance
        or float(np.sum(x)) > float(instance.k) + tolerance
    ):
        return None
    x = np.clip(x, 0.0, 1.0)
    z = _optimal_z(x, instance.k)
    if z is None:
        return None
    t = np.divide(
        x * x,
        z,
        out=np.zeros_like(x),
        where=z > 0.0,
    )
    exposure = _factor_exposure(instance, x)
    risk_epigraph = float(exposure @ exposure)
    return x, z, t, exposure, risk_epigraph


def _complete_solution_result(
    result: Dict[str, Any],
    instance: _PreparedInstance,
    x: np.ndarray,
    model_objective: float,
    postprocess_start: float,
    domain_tolerance: float,
) -> None:
    evaluation = _external_evaluation(
        instance,
        x,
        domain_tolerance=domain_tolerance,
    )
    diagnostics = evaluation.pop("_diagnostics")
    external_objective = evaluation["objective"]
    result.update(
        {
            "has_solution": True,
            "x": np.asarray(x, dtype=np.float64),
            "model_objective": float(model_objective),
            "external_objective": external_objective,
            "objective_scaling_gap": (
                float(model_objective) - float(external_objective)
                if external_objective is not None
                else None
            ),
            "evaluation": evaluation,
            **diagnostics,
            "postprocess_seconds": (
                time.perf_counter() - postprocess_start
            ),
            "restart_state": RelaxationState(
                backend=str(result.get("solver", "commercial")),
                implementation="python",
                dimension=instance.dimension,
                k=instance.k,
                constraint_ids=instance.constraint_names,
                x=np.asarray(x, dtype=np.float64),
                scalars={
                    "fresh_restart_epoch_recommended": True,
                },
            ).to_dict(copy=False),
        }
    )


__all__: list[str] = []
