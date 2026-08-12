"""Portable warm state for incumbent searches and future BnB nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np


@dataclass
class IncumbentState:
    dimension: int
    k: int
    constraint_ids: tuple[str, ...]
    x: np.ndarray
    selectors: np.ndarray
    objective: Optional[float] = None
    required_assets: tuple[int, ...] = ()
    forbidden_assets: tuple[int, ...] = ()
    sample_counter: int = 0

    def __post_init__(self) -> None:
        self.dimension = int(self.dimension)
        self.k = int(self.k)
        self.constraint_ids = tuple(str(value) for value in self.constraint_ids)
        self.x = np.asarray(self.x, dtype=float).reshape(-1)
        self.selectors = np.asarray(self.selectors, dtype=float).reshape(-1)
        self.required_assets = tuple(
            int(value) for value in self.required_assets
        )
        self.forbidden_assets = tuple(
            int(value) for value in self.forbidden_assets
        )
        self.sample_counter = int(self.sample_counter)
        if self.x.shape != (self.dimension,):
            raise ValueError("incumbent-state x has the wrong dimension")
        if self.selectors.shape != (self.dimension,):
            raise ValueError("incumbent-state selectors have the wrong dimension")

    @property
    def support(self) -> np.ndarray:
        return np.flatnonzero(self.selectors > 0.5)

    def to_dict(self, copy: bool = True) -> dict[str, Any]:
        clone = (lambda value: value.copy()) if copy else (lambda value: value)
        return {
            "dimension": self.dimension,
            "k": self.k,
            "constraint_ids": list(self.constraint_ids),
            "x": clone(self.x),
            "selectors": clone(self.selectors),
            "objective": self.objective,
            "required_assets": list(self.required_assets),
            "forbidden_assets": list(self.forbidden_assets),
            "sample_counter": self.sample_counter,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "IncumbentState":
        return cls(
            dimension=int(value["dimension"]),
            k=int(value["k"]),
            constraint_ids=tuple(value.get("constraint_ids", ())),
            x=value["x"],
            selectors=value["selectors"],
            objective=(
                None
                if value.get("objective") is None
                else float(value["objective"])
            ),
            required_assets=tuple(value.get("required_assets", ())),
            forbidden_assets=tuple(value.get("forbidden_assets", ())),
            sample_counter=int(value.get("sample_counter", 0)),
        )

    @classmethod
    def coerce(cls, value: Any) -> Optional["IncumbentState"]:
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        state = getattr(value, "state", None)
        if isinstance(state, cls):
            return state
        if isinstance(value, Mapping):
            payload = value.get("state", value)
            if isinstance(payload, cls):
                return payload
            if isinstance(payload, Mapping) and {
                "dimension",
                "k",
                "x",
                "selectors",
            }.issubset(payload):
                return cls.from_mapping(payload)
        return None

    def compatible_support(
        self,
        dimension: int,
        required_assets: Sequence[int] = (),
        forbidden_assets: Sequence[int] = (),
    ) -> np.ndarray:
        if int(dimension) != self.dimension:
            raise ValueError("warm incumbent dimension does not match")
        support = set(int(value) for value in self.support)
        support.update(int(value) for value in required_assets)
        support.difference_update(int(value) for value in forbidden_assets)
        return np.asarray(sorted(support), dtype=np.int64)


__all__ = ["IncumbentState"]
