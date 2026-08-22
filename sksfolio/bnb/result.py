"""Typed result returned by the sparse branch-and-bound solver."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterator, Mapping, Optional

import numpy as np


@dataclass
class BranchAndBoundResult(Mapping[str, Any]):
    raw: dict[str, Any]

    @property
    def status(self) -> str:
        return str(self.raw.get("status", "unknown"))

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
    def upper_bound(self) -> Optional[float]:
        value = self.raw.get("upper_bound")
        if value is None:
            return None
        result = float(value)
        return result if math.isfinite(result) else None

    @property
    def lower_bound(self) -> Optional[float]:
        value = self.raw.get("lower_bound")
        if value is None:
            return None
        result = float(value)
        return result if math.isfinite(result) else None

    @property
    def absolute_gap(self) -> Optional[float]:
        value = self.raw.get("absolute_gap")
        return None if value is None else float(value)

    @property
    def relative_gap(self) -> Optional[float]:
        value = self.raw.get("relative_gap")
        return None if value is None else float(value)

    @property
    def optimal(self) -> bool:
        return self.status == "optimal"

    @property
    def nodes_processed(self) -> int:
        return int(self.raw.get("nodes_processed", 0))

    @property
    def cuts(self) -> tuple[dict[str, Any], ...]:
        return tuple(self.raw.get("cuts", ()))

    def to_dict(self) -> dict[str, Any]:
        return dict(self.raw)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.raw)

    def __len__(self) -> int:
        return len(self.raw)


__all__ = ["BranchAndBoundResult"]
