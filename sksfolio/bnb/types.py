"""Small, solver-independent data objects used by branch-and-bound.

The branch-and-bound implementation represents binary selector fixings by
Python integers.  A bit set in ``fixed_one_mask`` means that the corresponding
selector is fixed to one; a bit set in ``fixed_zero_mask`` means that it is
fixed to zero.  Python integers are arbitrary precision, so this remains exact
for portfolios with more than 64 assets while keeping the common set
operations in optimized C code.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from operator import index as integer_index
from typing import Any, Iterable, Mapping, Optional


def indices_to_mask(indices: Iterable[int], dimension: int) -> int:
    """Return the exact integer bit mask for a sequence of asset indices."""
    size = integer_index(dimension)
    if size < 0:
        raise ValueError("dimension must be nonnegative")
    result = 0
    for raw_index in indices:
        asset = integer_index(raw_index)
        if asset < 0 or asset >= size:
            raise ValueError(f"asset index {asset} is outside [0, {size})")
        result |= 1 << asset
    return result


def mask_to_indices(mask: int) -> tuple[int, ...]:
    """Return the increasing indices represented by a nonnegative mask."""
    remaining = integer_index(mask)
    if remaining < 0:
        raise ValueError("a selector mask must be nonnegative")
    result: list[int] = []
    while remaining:
        bit = remaining & -remaining
        result.append(bit.bit_length() - 1)
        remaining ^= bit
    return tuple(result)


def validate_mask(mask: int, dimension: int, name: str) -> int:
    """Normalize a bit mask and verify that it fits the stated dimension."""
    value = integer_index(mask)
    size = integer_index(dimension)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    if size < 0:
        raise ValueError("dimension must be nonnegative")
    if value >> size:
        raise ValueError(f"{name} contains an index outside [0, {size})")
    return value


@dataclass(frozen=True, slots=True)
class BinaryFixings:
    """Immutable binary fixings at one branch-and-bound node."""

    dimension: int
    fixed_one_mask: int = 0
    fixed_zero_mask: int = 0

    def __post_init__(self) -> None:
        size = integer_index(self.dimension)
        one = validate_mask(self.fixed_one_mask, size, "fixed_one_mask")
        zero = validate_mask(self.fixed_zero_mask, size, "fixed_zero_mask")
        if one & zero:
            raise ValueError("a selector cannot be fixed to both zero and one")
        object.__setattr__(self, "dimension", size)
        object.__setattr__(self, "fixed_one_mask", one)
        object.__setattr__(self, "fixed_zero_mask", zero)

    @classmethod
    def from_indices(
        cls,
        dimension: int,
        *,
        fixed_one: Iterable[int] = (),
        fixed_zero: Iterable[int] = (),
    ) -> "BinaryFixings":
        """Construct node fixings from ordinary index collections."""
        return cls(
            dimension=dimension,
            fixed_one_mask=indices_to_mask(fixed_one, dimension),
            fixed_zero_mask=indices_to_mask(fixed_zero, dimension),
        )

    @property
    def fixed_one(self) -> tuple[int, ...]:
        return mask_to_indices(self.fixed_one_mask)

    @property
    def fixed_zero(self) -> tuple[int, ...]:
        return mask_to_indices(self.fixed_zero_mask)

    @property
    def fixed_mask(self) -> int:
        return self.fixed_one_mask | self.fixed_zero_mask

    @property
    def free_mask(self) -> int:
        return ((1 << self.dimension) - 1) & ~self.fixed_mask

    @property
    def free(self) -> tuple[int, ...]:
        return mask_to_indices(self.free_mask)

    @property
    def is_complete(self) -> bool:
        return self.fixed_mask.bit_count() == self.dimension

    def with_masks(self, fixed_one_mask: int, fixed_zero_mask: int) -> "BinaryFixings":
        """Return fixings with replacement masks and the same dimension."""
        return BinaryFixings(
            self.dimension,
            fixed_one_mask=fixed_one_mask,
            fixed_zero_mask=fixed_zero_mask,
        )

    def branch(self, asset: int) -> tuple["BinaryFixings", "BinaryFixings"]:
        """Return the ``z_asset = 1`` and ``z_asset = 0`` child fixings."""
        position = integer_index(asset)
        if position < 0 or position >= self.dimension:
            raise ValueError("branching asset is outside the node dimension")
        bit = 1 << position
        if self.fixed_mask & bit:
            raise ValueError("cannot branch on an already fixed selector")
        return (
            self.with_masks(self.fixed_one_mask | bit, self.fixed_zero_mask),
            self.with_masks(self.fixed_one_mask, self.fixed_zero_mask | bit),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "dimension": self.dimension,
            "fixed_one": list(self.fixed_one),
            "fixed_zero": list(self.fixed_zero),
        }


@dataclass(frozen=True, slots=True)
class MultiSelectorCut:
    """Public, serializable view of one certified forbidden pattern.

    The cut forbids every binary vector satisfying ``z_i = 1`` for every
    index in ``forced_one`` and ``z_i = 0`` for every index in
    ``forced_zero``.  Equivalently, it represents

    ``sum(forced_one, z_i) - sum(forced_zero, z_i) <= |forced_one| - 1``.
    """

    cut_id: int
    forced_one: tuple[int, ...]
    forced_zero: tuple[int, ...]
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    safety_margin: float = 0.0
    source: str = "safe_screening"

    @property
    def right_hand_side(self) -> int:
        return len(self.forced_one) - 1

    @property
    def forced_one_mask(self) -> int:
        """Exact mask containing every required-one literal in the pattern."""
        return sum(1 << asset for asset in self.forced_one)

    @property
    def forced_zero_mask(self) -> int:
        """Exact mask containing every required-zero literal in the pattern."""
        return sum(1 << asset for asset in self.forced_zero)

    @property
    def literal_count(self) -> int:
        return len(self.forced_one) + len(self.forced_zero)

    def excludes(self, selectors: Iterable[Any]) -> bool:
        """Return whether a binary selector vector matches this pattern."""
        values = tuple(selectors)
        required = self.forced_one + self.forced_zero
        if required and max(required) >= len(values):
            raise ValueError("selector vector is shorter than the cut dimension")
        return all(bool(values[i]) for i in self.forced_one) and all(
            not bool(values[i]) for i in self.forced_zero
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable public representation."""
        return {
            "cut_id": self.cut_id,
            "forced_one": list(self.forced_one),
            "forced_zero": list(self.forced_zero),
            "forced_one_mask": self.forced_one_mask,
            "forced_zero_mask": self.forced_zero_mask,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "safety_margin": self.safety_margin,
            "source": self.source,
            "right_hand_side": self.right_hand_side,
        }


@dataclass(frozen=True, slots=True)
class CutInsertionResult:
    """Outcome of attempting to insert a cut into a cut pool."""

    accepted: bool
    reason: str
    cut: Optional[MultiSelectorCut] = None
    dominating_cut: Optional[MultiSelectorCut] = None
    removed_cut_ids: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "cut": None if self.cut is None else self.cut.to_dict(),
            "dominating_cut": (
                None
                if self.dominating_cut is None
                else self.dominating_cut.to_dict()
            ),
            "removed_cut_ids": list(self.removed_cut_ids),
        }


@dataclass(frozen=True, slots=True)
class CutPropagationResult:
    """Closure of cut-based unit propagation at one node."""

    fixings: BinaryFixings
    infeasible: bool
    closed: bool
    newly_fixed_one_mask: int = 0
    newly_fixed_zero_mask: int = 0
    conflict_cut: Optional[MultiSelectorCut] = None
    rounds: int = 0
    cuts_examined: int = 0

    @property
    def newly_fixed_one(self) -> tuple[int, ...]:
        return mask_to_indices(self.newly_fixed_one_mask)

    @property
    def newly_fixed_zero(self) -> tuple[int, ...]:
        return mask_to_indices(self.newly_fixed_zero_mask)

    @property
    def fixed_count(self) -> int:
        return (
            self.newly_fixed_one_mask | self.newly_fixed_zero_mask
        ).bit_count()

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixings": self.fixings.to_dict(),
            "infeasible": self.infeasible,
            "closed": self.closed,
            "newly_fixed_one": list(self.newly_fixed_one),
            "newly_fixed_zero": list(self.newly_fixed_zero),
            "conflict_cut": (
                None if self.conflict_cut is None else self.conflict_cut.to_dict()
            ),
            "rounds": self.rounds,
            "cuts_examined": self.cuts_examined,
        }


@dataclass(frozen=True, slots=True)
class CutPoolStats:
    """Immutable statistics snapshot for a multi-selector cut pool."""

    active_cuts: int
    insertion_attempts: int
    accepted_cuts: int
    duplicate_rejections: int
    dominated_rejections: int
    uncertified_rejections: int
    removed_dominated_cuts: int
    propagation_calls: int
    cuts_examined: int
    unit_fixings: int
    conflicts: int

    def to_dict(self) -> dict[str, int]:
        return {
            field: int(getattr(self, field))
            for field in self.__dataclass_fields__
        }


def finite_or_none(value: Any, name: str) -> Optional[float]:
    """Normalize optional evidence values used in public cut records."""
    if value is None:
        return None
    result = float(value)
    if math.isnan(result):
        raise ValueError(f"{name} cannot be NaN")
    return result


__all__ = [
    "BinaryFixings",
    "CutInsertionResult",
    "CutPoolStats",
    "CutPropagationResult",
    "MultiSelectorCut",
    "indices_to_mask",
    "mask_to_indices",
]
