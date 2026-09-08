"""Selector-aware perspective primitives for branch-and-bound nodes.

At a node, let ``I`` contain selectors fixed to one, ``O`` selectors fixed
to zero, and ``F`` the remaining free selectors.  The perspective term is

``0.5 * sum_{i in I} x_i**2 + G_{k-|I|}(x_F)``

with ``x_O = 0``.  Consequently its proximal map is a clipped ridge step on
``I``, zero on ``O``, and the existing PAVA oracle on ``F``.  The conjugate
has the matching fixed-coordinate sum plus a top-``k-|I|`` free-coordinate
sum.  Keeping both operations here prevents node relaxations and safe dual
bounds from silently using different selector semantics.
"""

from __future__ import annotations

import math
from operator import index as integer_index
from typing import Any, Iterable

import numpy as np

from ..relaxation.problem import perspective_value
from ..relaxation.pava import prox as pava_prox


def _indices(
    values: Iterable[int],
    dimension: int,
    name: str,
) -> np.ndarray:
    normalized: list[int] = []
    for raw in values:
        if isinstance(raw, (bool, np.bool_)):
            raise TypeError(f"{name} must contain integer indices")
        try:
            value = integer_index(raw)
        except TypeError as error:
            raise TypeError(
                f"{name} must contain integer indices"
            ) from error
        if value < 0 or value >= dimension:
            raise ValueError(
                f"{name} contains an index outside [0, {dimension})"
            )
        normalized.append(value)
    if not normalized:
        return np.empty(0, dtype=np.int64)
    return np.asarray(sorted(set(normalized)), dtype=np.int64)


def _pava_method(value: str) -> str:
    normalized = str(value).lower().replace("-", "_")
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


def _coordinate_scores(argument: np.ndarray, weight: float) -> np.ndarray:
    """Return conjugate scores for one boxed quadratic coordinate."""
    result = np.zeros_like(argument)
    quadratic = (argument > 0.0) & (argument < weight)
    linear = argument >= weight
    with np.errstate(over="ignore", invalid="ignore"):
        result[quadratic] = (
            0.5
            * argument[quadratic]
            * (argument[quadratic] / weight)
        )
    result[linear] = argument[linear] - 0.5 * weight
    return result


def _deterministic_largest(
    scores: np.ndarray,
    indices: np.ndarray,
    count: int,
) -> np.ndarray:
    """Select largest scores, resolving every boundary tie by index."""
    selected_count = min(max(int(count), 0), indices.size)
    if selected_count == 0:
        return np.empty(0, dtype=np.int64)
    if selected_count == indices.size:
        return indices.copy()
    values = scores[indices]
    split = values.size - selected_count
    threshold = float(np.partition(values, split)[split])
    strict = indices[values > threshold]
    tied = np.sort(indices[values == threshold])
    remaining = selected_count - strict.size
    selected = np.concatenate((strict, tied[:remaining]))
    # Returning increasing indices makes results reproducible across the
    # full- and partial-selection paths without changing the selected sum.
    return np.sort(selected).astype(np.int64, copy=False)


class SelectorPerspectiveOracle:
    """Perspective value, conjugate, and prox under binary node fixings.

    ``forced_one`` consumes selector capacity but does not require a positive
    portfolio weight.  This distinction is essential: replacing ``z_i = 1``
    by a constraint on ``x_i`` would describe a different problem.
    """

    def __init__(
        self,
        dimension: int,
        k: int,
        *,
        forced_one: Iterable[int] = (),
        forced_zero: Iterable[int] = (),
        pava_method: str = "partial_sort",
    ) -> None:
        if isinstance(dimension, (bool, np.bool_)):
            raise TypeError("dimension must be an integer")
        if isinstance(k, (bool, np.bool_)):
            raise TypeError("k must be an integer")
        try:
            size = integer_index(dimension)
            cardinality = integer_index(k)
        except TypeError as error:
            raise TypeError("dimension and k must be integers") from error
        if size < 1:
            raise ValueError("dimension must be positive")
        if cardinality < 1 or cardinality > size:
            raise ValueError("k must lie in {1, ..., dimension}")

        one = _indices(forced_one, size, "forced_one")
        zero = _indices(forced_zero, size, "forced_zero")
        if np.intersect1d(one, zero, assume_unique=True).size:
            raise ValueError(
                "forced_one and forced_zero must be disjoint"
            )
        if one.size > cardinality:
            raise ValueError("more than k selectors are fixed to one")

        free_mask = np.ones(size, dtype=bool)
        free_mask[one] = False
        free_mask[zero] = False
        free = np.flatnonzero(free_mask).astype(np.int64, copy=False)

        self.dimension = size
        self.k = cardinality
        self.forced_one = one
        self.forced_zero = zero
        self.free_indices = free
        self.remaining_capacity = cardinality - one.size
        self.effective_capacity = min(
            self.remaining_capacity,
            free.size,
        )
        self.pava_method = _pava_method(pava_method)

        # Fixing arrays are structural data.  Read-only views prevent an
        # accidental in-place mutation from invalidating cached capacities.
        self.forced_one.setflags(write=False)
        self.forced_zero.setflags(write=False)
        self.free_indices.setflags(write=False)

    @property
    def has_fixings(self) -> bool:
        return bool(self.forced_one.size or self.forced_zero.size)

    def _vector(
        self,
        value: Any,
        name: str,
        *,
        require_finite: bool,
    ) -> np.ndarray:
        result = np.asarray(value, dtype=np.float64).reshape(-1)
        if result.shape != (self.dimension,):
            raise ValueError(f"{name} has the wrong dimension")
        if require_finite and np.any(~np.isfinite(result)):
            raise ValueError(f"{name} must be finite")
        return result

    @staticmethod
    def _positive_finite(value: float, name: str) -> float:
        result = float(value)
        if not math.isfinite(result) or result <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
        return result

    def prox(self, argument: Any, gamma: float) -> np.ndarray:
        """Return the exact node proximal point using the selected PAVA."""
        values = self._vector(
            argument,
            "argument",
            require_finite=True,
        )
        scale = self._positive_finite(gamma, "gamma")
        result = np.zeros(self.dimension, dtype=np.float64)

        if self.forced_one.size:
            result[self.forced_one] = np.clip(
                values[self.forced_one] / (1.0 + scale),
                0.0,
                1.0,
            )

        free = self.free_indices
        capacity = self.effective_capacity
        if free.size and capacity > 0:
            if capacity == free.size:
                # This is the separable k=d endpoint of PAVA.  Evaluating it
                # directly also covers the case where raw remaining capacity
                # exceeds the number of free coordinates.
                result[free] = np.clip(
                    values[free] / (1.0 + scale),
                    0.0,
                    1.0,
                )
            else:
                result[free] = pava_prox(
                    values[free],
                    scale,
                    capacity,
                    self.pava_method,
                )
        return result

    def coordinate_scores(
        self,
        argument: Any,
        weight: float,
    ) -> np.ndarray:
        """Return all boxed-coordinate conjugate scores."""
        values = self._vector(
            argument,
            "argument",
            require_finite=False,
        )
        scale = self._positive_finite(weight, "weight")
        if np.any(~np.isfinite(values)):
            return np.full(self.dimension, math.inf)
        return _coordinate_scores(values, scale)

    def selected_free_indices(
        self,
        argument: Any,
        weight: float,
    ) -> np.ndarray:
        """Return deterministic top-capacity free conjugate coordinates."""
        scores = self.coordinate_scores(argument, weight)
        if np.any(~np.isfinite(scores)):
            raise ValueError(
                "conjugate scores overflowed; rescale the objective"
            )
        return _deterministic_largest(
            scores,
            self.free_indices,
            self.effective_capacity,
        )

    def conjugate(self, argument: Any, weight: float) -> float:
        """Evaluate the conjugate of ``weight`` times the node perspective."""
        scores = self.coordinate_scores(argument, weight)
        if np.any(~np.isfinite(scores)):
            return math.inf
        selected = _deterministic_largest(
            scores,
            self.free_indices,
            self.effective_capacity,
        )
        return float(
            np.sum(scores[self.forced_one], dtype=np.float64)
            + np.sum(scores[selected], dtype=np.float64)
        )

    def domain_violation(self, x: Any) -> float:
        """Return the maximum violation of the node perspective domain."""
        vector = self._vector(x, "x", require_finite=False)
        if np.any(~np.isfinite(vector)):
            return math.inf
        violation = max(
            0.0,
            -float(np.min(vector)),
            float(np.max(vector)) - 1.0,
        )
        if self.forced_zero.size:
            violation = max(
                violation,
                float(np.max(np.abs(vector[self.forced_zero]))),
            )
        if self.free_indices.size:
            free_mass = float(
                np.sum(
                    np.maximum(vector[self.free_indices], 0.0),
                    dtype=np.float64,
                )
            )
            if not math.isfinite(free_mass):
                return math.inf
            violation = max(
                violation,
                free_mass - float(self.remaining_capacity),
            )
        return max(0.0, float(violation))

    def value(self, x: Any, tolerance: float = 1e-8) -> float:
        """Evaluate the node perspective, returning infinity off-domain."""
        vector = self._vector(x, "x", require_finite=False)
        active_tolerance = float(tolerance)
        if (
            not math.isfinite(active_tolerance)
            or active_tolerance < 0.0
        ):
            raise ValueError("tolerance must be finite and nonnegative")
        if np.any(~np.isfinite(vector)):
            return math.inf
        if self.domain_violation(vector) > active_tolerance:
            return math.inf

        fixed_value = 0.5 * float(
            vector[self.forced_one] @ vector[self.forced_one]
        )
        free = self.free_indices
        capacity = self.effective_capacity
        if not free.size or capacity == 0:
            return fixed_value
        free_value = perspective_value(
            vector[free],
            capacity,
            tolerance=active_tolerance,
        )
        return float(fixed_value + free_value)


__all__ = ["SelectorPerspectiveOracle"]
