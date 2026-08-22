"""Result object returned by the public incumbent API."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterator, Mapping, Optional

import numpy as np

from .state import IncumbentState


@dataclass
class IncumbentResult(Mapping[str, Any]):
    raw: dict[str, Any]

    @property
    def weights(self) -> Optional[np.ndarray]:
        value = self.raw.get("x")
        return None if value is None else np.asarray(value, dtype=float).reshape(-1)

    @property
    def selectors(self) -> Optional[np.ndarray]:
        value = self.raw.get("selectors")
        return None if value is None else np.asarray(value, dtype=float).reshape(-1)

    @property
    def support(self) -> np.ndarray:
        selectors = self.selectors
        if selectors is None:
            return np.empty(0, dtype=np.int64)
        return np.flatnonzero(selectors > 0.5)

    @property
    def active_support(self) -> np.ndarray:
        weights = self.weights
        if weights is None:
            return np.empty(0, dtype=np.int64)
        tolerance = float(self.raw.get("active_tolerance", 1e-9))
        return np.flatnonzero(np.abs(weights) > tolerance)

    @property
    def upper_bound(self) -> Optional[float]:
        value = self.raw.get("upper_bound")
        if value is None:
            return None
        result = float(value)
        return result if math.isfinite(result) else None

    @property
    def feasible(self) -> bool:
        return bool(self.raw.get("numerically_feasible", False))

    @property
    def status(self) -> str:
        return str(self.raw.get("status", "unknown"))

    @property
    def state(self) -> Optional[IncumbentState]:
        value = self.raw.get("state")
        return IncumbentState.coerce(value)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.raw)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.raw)

    def __len__(self) -> int:
        return len(self.raw)


__all__ = ["IncumbentResult"]
