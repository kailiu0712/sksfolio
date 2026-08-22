"""Portable warm-start state shared by every relaxation backend.

The state is deliberately a *restart* state rather than a promise of
bit-for-bit continuation.  That distinction matters for branch-and-bound:
when a child node adds or removes rows, dual variables are matched by stable
constraint names and the acceleration epoch is restarted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import numpy as np


STATE_SCHEMA_VERSION = 1
PathLike = Union[str, Path]


def _vector(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains a nonfinite value")
    return array.copy()


def _json_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (list, tuple)):
        return [_json_scalar(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_scalar(item) for key, item in value.items()}
    raise TypeError(
        "state scalars must contain only JSON-compatible values"
    )


@dataclass
class RelaxationState:
    """Versioned, cross-backend restart state.

    Dual multipliers in the public fields use the original problem's
    coordinates.  Backend-specific vectors and scalar tuning information can
    be stored in ``arrays`` and ``scalars``.  New branch rows are initialized
    with zero multipliers when :meth:`mapped_constraint_dual` is used.
    """

    backend: str
    implementation: str
    dimension: int
    k: int
    constraint_ids: Tuple[str, ...]
    x: np.ndarray
    factor_dual: Optional[np.ndarray] = None
    constraint_dual: Optional[np.ndarray] = None
    arrays: Dict[str, np.ndarray] = field(default_factory=dict)
    scalars: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.backend = str(self.backend)
        self.implementation = str(self.implementation)
        self.dimension = int(self.dimension)
        self.k = int(self.k)
        self.schema_version = int(self.schema_version)
        self.constraint_ids = tuple(str(item) for item in self.constraint_ids)
        self.x = _vector(self.x, "x")
        self.factor_dual = (
            None
            if self.factor_dual is None
            else _vector(self.factor_dual, "factor_dual")
        )
        self.constraint_dual = (
            None
            if self.constraint_dual is None
            else _vector(self.constraint_dual, "constraint_dual")
        )
        self.arrays = {
            str(key): _vector(value, f"arrays[{key!r}]")
            for key, value in dict(self.arrays).items()
        }
        self.scalars = {
            str(key): _json_scalar(value)
            for key, value in dict(self.scalars).items()
        }
        self.validate()

    def validate(self) -> None:
        if self.schema_version != STATE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported state schema version {self.schema_version}"
            )
        if self.dimension < 1:
            raise ValueError("state dimension must be positive")
        if not 1 <= self.k <= self.dimension:
            raise ValueError("state k must lie in {1, ..., dimension}")
        if self.x.shape != (self.dimension,):
            raise ValueError("state x has the wrong dimension")
        if len(set(self.constraint_ids)) != len(self.constraint_ids):
            raise ValueError(
                "constraint identifiers must be unique for warm starts"
            )
        if (
            self.constraint_dual is not None
            and self.constraint_dual.shape != (len(self.constraint_ids),)
        ):
            raise ValueError(
                "constraint_dual must match constraint identifiers"
            )

    def compatible_primal(self, instance: Any) -> np.ndarray:
        """Return a copy of the primal start after structural checks."""
        if int(instance.dimension) != self.dimension:
            raise ValueError(
                "warm-start dimension does not match the problem"
            )
        if int(instance.k) != self.k:
            raise ValueError("warm-start k does not match the problem")
        return self.x.copy()

    def mapped_constraint_dual(self, instance: Any) -> Optional[np.ndarray]:
        """Map original-coordinate row multipliers by constraint name."""
        if self.constraint_dual is None:
            return None
        names = tuple(str(item) for item in instance.constraint_names)
        if len(set(names)) != len(names):
            raise ValueError(
                "problem constraint names must be unique for warm starts"
            )
        lookup = dict(zip(self.constraint_ids, self.constraint_dual))
        return np.asarray(
            [float(lookup.get(name, 0.0)) for name in names],
            dtype=float,
        )

    def to_dict(self, *, copy: bool = True) -> Dict[str, Any]:
        clone = (lambda value: value.copy()) if copy else (lambda value: value)
        return {
            "schema_version": int(self.schema_version),
            "backend": self.backend,
            "implementation": self.implementation,
            "dimension": int(self.dimension),
            "k": int(self.k),
            "constraint_ids": list(self.constraint_ids),
            "x": clone(self.x),
            "factor_dual": (
                None
                if self.factor_dual is None
                else clone(self.factor_dual)
            ),
            "constraint_dual": (
                None
                if self.constraint_dual is None
                else clone(self.constraint_dual)
            ),
            "arrays": {
                key: clone(value) for key, value in self.arrays.items()
            },
            "scalars": dict(self.scalars),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RelaxationState":
        return cls(
            schema_version=int(
                value.get("schema_version", STATE_SCHEMA_VERSION)
            ),
            backend=str(value.get("backend", "unknown")),
            implementation=str(value.get("implementation", "python")),
            dimension=int(value["dimension"]),
            k=int(value["k"]),
            constraint_ids=tuple(value.get("constraint_ids", ())),
            x=value["x"],
            factor_dual=value.get("factor_dual"),
            constraint_dual=value.get("constraint_dual"),
            arrays=dict(value.get("arrays", {})),
            scalars=dict(value.get("scalars", {})),
        )

    @classmethod
    def coerce(cls, value: Any) -> Optional["RelaxationState"]:
        if value is None or value is False:
            return None
        if isinstance(value, cls):
            return value
        if hasattr(value, "state"):
            candidate = value.state
            if candidate is not None:
                return cls.coerce(candidate)
        if isinstance(value, Mapping):
            if "restart_state" in value:
                return cls.coerce(value["restart_state"])
            if "state" in value:
                return cls.coerce(value["state"])
            return cls.from_mapping(value)
        raise TypeError(
            "warm_start must be a RelaxationState, result, mapping, or None"
        )

    def save(self, path: PathLike) -> Path:
        """Save state as a non-pickle NPZ archive."""
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": int(self.schema_version),
            "backend": self.backend,
            "implementation": self.implementation,
            "dimension": int(self.dimension),
            "k": int(self.k),
            "constraint_ids": list(self.constraint_ids),
            "has_factor_dual": self.factor_dual is not None,
            "has_constraint_dual": self.constraint_dual is not None,
            "array_keys": sorted(self.arrays),
            "scalars": self.scalars,
        }
        arrays: Dict[str, Any] = {
            "metadata": np.asarray(json.dumps(metadata)),
            "x": self.x,
        }
        if self.factor_dual is not None:
            arrays["factor_dual"] = self.factor_dual
        if self.constraint_dual is not None:
            arrays["constraint_dual"] = self.constraint_dual
        for key, value in self.arrays.items():
            arrays[f"payload_{key}"] = value
        with target.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        return target

    @classmethod
    def load(cls, path: PathLike) -> "RelaxationState":
        source = Path(path).expanduser().resolve()
        with np.load(source, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            arrays = {
                key: np.asarray(archive[f"payload_{key}"], dtype=float)
                for key in metadata.get("array_keys", ())
            }
            return cls(
                schema_version=int(metadata["schema_version"]),
                backend=str(metadata["backend"]),
                implementation=str(metadata["implementation"]),
                dimension=int(metadata["dimension"]),
                k=int(metadata["k"]),
                constraint_ids=tuple(metadata["constraint_ids"]),
                x=np.asarray(archive["x"], dtype=float),
                factor_dual=(
                    np.asarray(archive["factor_dual"], dtype=float)
                    if metadata.get("has_factor_dual")
                    else None
                ),
                constraint_dual=(
                    np.asarray(archive["constraint_dual"], dtype=float)
                    if metadata.get("has_constraint_dual")
                    else None
                ),
                arrays=arrays,
                scalars=dict(metadata.get("scalars", {})),
            )


__all__ = ["RelaxationState", "STATE_SCHEMA_VERSION"]
