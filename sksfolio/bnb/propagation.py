"""Cheap row-wise feasibility bounds for sparse BnB nodes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bounds import indices_from_mask


@dataclass(frozen=True)
class RowFeasibilityResult:
    feasible: bool
    maximum_lower_shortfall: float
    maximum_upper_excess: float
    row_lower_envelope: np.ndarray
    row_upper_envelope: np.ndarray


class RowFeasibilityOracle:
    """Bound each row over every support compatible with one node.

    The calculation uses ``0 <= x_i <= z_i`` and the remaining cardinality
    capacity. It is a necessary feasibility test, not a relaxation solve.
    """

    def __init__(
        self,
        instance,
        active_mask: int,
        *,
        maximum_dense_entries: int = 5_000_000,
    ) -> None:
        self.instance = instance
        self.dimension = int(instance.dimension)
        self.rows = int(instance.rows)
        self.k = int(instance.k)
        self.active_mask = int(active_mask)
        self.active_indices = indices_from_mask(self.active_mask)
        entries = self.rows * self.active_indices.size
        self.enabled = bool(entries <= int(maximum_dense_entries))
        self.storage = "disabled_oversized"
        self.coefficients = None
        if self.rows == 0:
            self.enabled = True
            self.storage = "empty"
            self.coefficients = np.empty((0, self.active_indices.size))
        elif self.enabled:
            selected = instance.C[:, self.active_indices]
            self.coefficients = np.asarray(
                selected.toarray() if hasattr(selected, "toarray") else selected,
                dtype=float,
            )
            self.storage = "dense_active_columns"
        self.lower = np.asarray(instance.lower, dtype=float).reshape(-1)
        self.upper = np.asarray(instance.upper, dtype=float).reshape(-1)

    @staticmethod
    def _top_sum(values: np.ndarray, count: int) -> np.ndarray:
        if values.shape[1] == 0 or count <= 0:
            return np.zeros(values.shape[0])
        selected = min(int(count), values.shape[1])
        if selected == values.shape[1]:
            return np.sum(values, axis=1)
        split = values.shape[1] - selected
        return np.sum(np.partition(values, split, axis=1)[:, split:], axis=1)

    def evaluate(
        self,
        fixed_one_mask: int,
        fixed_zero_mask: int,
        *,
        tolerance: float = 1e-10,
    ) -> RowFeasibilityResult:
        if not self.enabled:
            return RowFeasibilityResult(
                True,
                0.0,
                0.0,
                np.full(self.rows, -np.inf),
                np.full(self.rows, np.inf),
            )
        if self.rows == 0:
            return RowFeasibilityResult(
                True,
                0.0,
                0.0,
                np.empty(0),
                np.empty(0),
            )
        one = int(fixed_one_mask)
        zero = int(fixed_zero_mask)
        if one & zero or one.bit_count() > self.k:
            return RowFeasibilityResult(
                False,
                np.inf,
                np.inf,
                np.full(self.rows, np.inf),
                np.full(self.rows, -np.inf),
            )
        fixed_one = np.fromiter(
            (bool(one & (1 << int(i))) for i in self.active_indices),
            dtype=bool,
            count=self.active_indices.size,
        )
        fixed_zero = np.fromiter(
            (bool(zero & (1 << int(i))) for i in self.active_indices),
            dtype=bool,
            count=self.active_indices.size,
        )
        free = ~(fixed_one | fixed_zero)
        capacity = self.k - one.bit_count()
        fixed_values = self.coefficients[:, fixed_one]
        free_values = self.coefficients[:, free]
        maximum = (
            np.sum(np.maximum(fixed_values, 0.0), axis=1)
            if fixed_values.shape[1]
            else np.zeros(self.rows)
        )
        minimum = (
            np.sum(np.minimum(fixed_values, 0.0), axis=1)
            if fixed_values.shape[1]
            else np.zeros(self.rows)
        )
        if free_values.shape[1] and capacity > 0:
            maximum += self._top_sum(
                np.maximum(free_values, 0.0),
                capacity,
            )
            minimum -= self._top_sum(
                np.maximum(-free_values, 0.0),
                capacity,
            )
        finite_lower = np.isfinite(self.lower)
        finite_upper = np.isfinite(self.upper)
        lower_shortfall = np.where(
            finite_lower,
            self.lower - maximum,
            -np.inf,
        )
        upper_excess = np.where(
            finite_upper,
            minimum - self.upper,
            -np.inf,
        )
        maximum_lower = (
            max(0.0, float(np.max(lower_shortfall)))
            if lower_shortfall.size
            else 0.0
        )
        maximum_upper = (
            max(0.0, float(np.max(upper_excess)))
            if upper_excess.size
            else 0.0
        )
        feasible = bool(
            maximum_lower <= float(tolerance)
            and maximum_upper <= float(tolerance)
        )
        return RowFeasibilityResult(
            feasible,
            maximum_lower,
            maximum_upper,
            minimum,
            maximum,
        )


__all__ = ["RowFeasibilityOracle", "RowFeasibilityResult"]
