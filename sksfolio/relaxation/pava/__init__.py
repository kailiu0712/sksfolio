"""Dispatch and cross-checks for the perspective PAVA primitives."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .full_sort import prox as prox_full_sort
from .partial_sort import (
    native_available as native_partial_sort_available,
    native_enabled as native_partial_sort_enabled,
    prox as prox_partial_sort,
    prox_native as prox_native_partial_sort,
    prox_python as prox_python_partial_sort,
)


PAVA_METHODS = ("full_sort", "partial_sort")


def prox(
    argument: Any,
    gamma: float,
    k: int,
    method: str = "partial_sort",
) -> np.ndarray:
    """Dispatch to a full-sort or partial-sort PAVA oracle."""
    normalized = str(method).lower().replace("-", "_")
    aliases = {
        "full": "full_sort",
        "partial": "partial_sort",
        "topk": "partial_sort",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized == "full_sort":
        return prox_full_sort(argument, gamma, k)
    if normalized == "partial_sort":
        return prox_partial_sort(argument, gamma, k)
    raise ValueError(
        "PAVA method must be 'full_sort' or 'partial_sort'"
    )


def check_pava_oracles(
    seed: int = 917,
    repetitions: int = 20,
) -> Dict[str, float]:
    """Cross-check partial selection against the full-sort implementation."""
    rng = np.random.default_rng(seed)
    maximum_error = 0.0
    cases = 0
    for dimension in (2, 5, 20, 100, 1000):
        selected_values = {
            1,
            max(1, dimension // 5),
            max(1, dimension - 1),
            dimension,
        }
        for k in sorted(selected_values):
            for _ in range(repetitions):
                argument = (
                    rng.normal(size=dimension)
                    * rng.uniform(0.1, 8.0)
                )
                gamma = float(rng.uniform(1e-4, 5.0))
                partial = prox_partial_sort(argument, gamma, k)
                full = prox_full_sort(argument, gamma, k)
                maximum_error = max(
                    maximum_error,
                    float(np.max(np.abs(partial - full))),
                )
                cases += 1
    return {
        "maximum_absolute_error": maximum_error,
        "seed": float(seed),
        "cases": float(cases),
    }

__all__ = [
    "PAVA_METHODS",
    "check_pava_oracles",
    "native_partial_sort_available",
    "native_partial_sort_enabled",
    "prox",
    "prox_full_sort",
    "prox_native_partial_sort",
    "prox_partial_sort",
    "prox_python_partial_sort",
]
