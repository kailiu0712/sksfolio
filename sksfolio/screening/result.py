"""Typed results for safe variable screening and support cuts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class ScreeningCut:
    """A safe no-good cut for one prescribed binary support pattern."""

    forced_one: tuple[int, ...]
    forced_zero: tuple[int, ...]
    lower_bound: float
    upper_bound: float
    valid: bool
    safety_margin: float = 0.0

    @property
    def right_hand_side(self) -> int:
        # Expanded form:
        # sum_{S} z_i - sum_{N} z_i <= |S| - 1.
        return len(self.forced_one) - 1

    def coefficients(self, dimension: int) -> np.ndarray:
        values = np.zeros(int(dimension), dtype=float)
        values[list(self.forced_one)] = 1.0
        values[list(self.forced_zero)] = -1.0
        return values

    def excludes(self, selectors: Any, tolerance: float = 1e-9) -> bool:
        """Return whether the selector violates this no-good cut."""
        vector = np.asarray(selectors, dtype=float).reshape(-1)
        left = float(np.sum(vector[list(self.forced_one)]))
        left -= float(np.sum(vector[list(self.forced_zero)]))
        return left > self.right_hand_side + float(tolerance)


@dataclass
class SafeScreeningResult(Mapping[str, Any]):
    raw: dict[str, Any]

    @property
    def fixed_zero(self) -> np.ndarray:
        return np.asarray(
            self.raw.get("fixed_zero", ()),
            dtype=np.int64,
        ).reshape(-1)

    @property
    def fixed_one(self) -> np.ndarray:
        return np.asarray(
            self.raw.get("fixed_one", ()),
            dtype=np.int64,
        ).reshape(-1)

    @property
    def fixed_out(self) -> np.ndarray:
        """Alias for selectors fixed to zero."""
        return self.fixed_zero

    @property
    def fixed_in(self) -> np.ndarray:
        """Alias for selectors fixed to one."""
        return self.fixed_one

    @property
    def newly_fixed_zero(self) -> np.ndarray:
        return np.asarray(
            self.raw.get("newly_fixed_zero", ()),
            dtype=np.int64,
        ).reshape(-1)

    @property
    def newly_fixed_one(self) -> np.ndarray:
        return np.asarray(
            self.raw.get("newly_fixed_one", ()),
            dtype=np.int64,
        ).reshape(-1)

    @property
    def prunable(self) -> bool:
        return bool(self.raw.get("prunable", False))

    @property
    def node_prunable(self) -> bool:
        return self.prunable

    @property
    def node_lower_bound(self) -> Optional[float]:
        value = self.raw.get("node_lower_bound")
        return None if value is None else float(value)

    @property
    def remaining_free(self) -> np.ndarray:
        return np.asarray(
            self.raw.get("remaining_free", ()),
            dtype=np.int64,
        ).reshape(-1)

    @property
    def include_lower_bounds(self) -> np.ndarray:
        """Conditional bounds for branches with ``z_i = 1``."""
        return np.asarray(
            self.raw.get("lower_bound_if_one", ()),
            dtype=float,
        ).reshape(-1)

    @property
    def exclude_lower_bounds(self) -> np.ndarray:
        """Conditional bounds for branches with ``z_i = 0``."""
        return np.asarray(
            self.raw.get("lower_bound_if_zero", ()),
            dtype=float,
        ).reshape(-1)

    @property
    def screened_count(self) -> int:
        return int(self.raw.get("screened_count", 0))

    @property
    def screened_fraction(self) -> float:
        return float(self.raw.get("screened_fraction", 0.0))

    @property
    def fixings(self) -> np.ndarray:
        """Return ``-1`` for fixed-out, ``1`` for fixed-in, else ``0``."""
        size = self.include_lower_bounds.size
        values = np.zeros(size, dtype=np.int8)
        values[self.fixed_zero] = -1
        values[self.fixed_one] = 1
        return values

    def to_dict(self) -> dict[str, Any]:
        return dict(self.raw)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.raw)

    def __len__(self) -> int:
        return len(self.raw)


__all__ = ["SafeScreeningResult", "ScreeningCut"]
