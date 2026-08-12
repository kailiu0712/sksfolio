"""Shared validation and scalar formulas for the two PAVA oracles."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


# Above this scale, subtracting the unit box cap can lose more than roughly
# sqrt(machine epsilon) in Float64.  The algorithms are intended to run after
# the PDHG objective normalization, so rejecting such inputs is preferable to
# returning silently inconsistent full- and partial-sort answers.
_FLOAT64_SAFE_SCALE = 1.0 / math.sqrt(np.finfo(np.float64).eps)
_FLOAT64_MIN_GAMMA = math.sqrt(np.finfo(np.float64).tiny)


def validate(
    argument: Any,
    gamma: float,
    k: int,
) -> np.ndarray:
    """Return a Float64 vector after common domain and scale checks."""
    values = np.asarray(argument, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("argument must be nonempty")
    if np.any(~np.isfinite(values)):
        raise ValueError("argument must be finite")
    if not math.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("gamma must be positive and finite")
    if isinstance(k, (bool, np.bool_)):
        raise ValueError("k must be an integer")
    try:
        selected = int(k)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("k must be an integer") from error
    if selected != k:
        raise ValueError("k must be an integer")
    if not 1 <= selected <= values.size:
        raise ValueError("k must lie in {1, ..., dimension}")

    maximum = float(np.max(np.abs(values)))
    if (
        maximum > _FLOAT64_SAFE_SCALE
        or gamma > _FLOAT64_SAFE_SCALE
        or gamma < _FLOAT64_MIN_GAMMA
    ):
        raise ValueError(
            "PAVA input is outside the conservative Float64 supported "
            "range; rescale the objective and data before calling the prox"
        )
    return values


def singleton_subtraction(
    values: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Return the uncoupled top-coordinate subtraction stably."""
    ratio = gamma / (1.0 + gamma)
    return np.where(
        values <= 1.0 + gamma,
        ratio * values,
        values - 1.0,
    )


__all__ = ["singleton_subtraction", "validate"]
