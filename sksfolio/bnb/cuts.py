"""Dominance-aware multi-selector no-good cuts for branch-and-bound.

A cut stores a forbidden partial binary assignment ``(S, N)``:

``z_i = 1 for i in S`` and ``z_i = 0 for i in N``.

If all but one literal of that pattern hold at a node, the last literal must
be reversed.  If every literal holds, the node is infeasible.  Both operations
use arbitrary-precision Python integer masks, making them exact and cheap even
when the number of selectors exceeds the machine word size.

Safety is deliberately explicit.  A direct insertion must either provide
valid bound evidence ``lower_bound > upper_bound + safety_margin`` or set
``certified=True`` to assert that the forbidden pattern was proved elsewhere.
The convenience method :meth:`add_screening_cut` accepts the existing
``ScreeningCut`` object only when its ``valid`` flag is true.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from operator import index as integer_index
from threading import RLock
from typing import Any, Iterable, Iterator, Optional

from .types import (
    BinaryFixings,
    CutInsertionResult,
    CutPoolStats,
    CutPropagationResult,
    MultiSelectorCut,
    finite_or_none,
    indices_to_mask,
    mask_to_indices,
    validate_mask,
)


@dataclass(frozen=True, slots=True)
class _StoredCut:
    one_mask: int
    zero_mask: int
    view: MultiSelectorCut


def _subsumes(left: _StoredCut, right_one: int, right_zero: int) -> bool:
    """Return whether ``left`` forbids every vector forbidden by ``right``."""
    return not (left.one_mask & ~right_one) and not (
        left.zero_mask & ~right_zero
    )


class MultiSelectorCutPool:
    """Thread-safe antichain of certified binary forbidden-pattern cuts.

    A shorter pattern dominates every longer pattern containing all its
    literals.  Insertions therefore retain only an antichain: redundant new
    cuts are rejected and existing cuts dominated by a new one are removed.
    This both reduces storage and makes propagation faster.
    """

    def __init__(self, dimension: int) -> None:
        size = integer_index(dimension)
        if size < 0:
            raise ValueError("dimension must be nonnegative")
        self.dimension = size
        self._cuts: dict[int, _StoredCut] = {}
        self._patterns: dict[tuple[int, int], int] = {}
        self._next_id = 0
        self._lock = RLock()
        self._insertion_attempts = 0
        self._accepted_cuts = 0
        self._duplicate_rejections = 0
        self._dominated_rejections = 0
        self._uncertified_rejections = 0
        self._removed_dominated_cuts = 0
        self._propagation_calls = 0
        self._cuts_examined = 0
        self._unit_fixings = 0
        self._conflicts = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._cuts)

    def __iter__(self) -> Iterator[MultiSelectorCut]:
        # Iterate over an immutable snapshot so callers never hold the lock.
        return iter(self.cuts)

    @property
    def cuts(self) -> tuple[MultiSelectorCut, ...]:
        """Return active cuts in increasing insertion-ID order."""
        with self._lock:
            return tuple(stored.view for stored in self._cuts.values())

    @property
    def stats(self) -> CutPoolStats:
        """Return an immutable, internally consistent statistics snapshot."""
        with self._lock:
            return CutPoolStats(
                active_cuts=len(self._cuts),
                insertion_attempts=self._insertion_attempts,
                accepted_cuts=self._accepted_cuts,
                duplicate_rejections=self._duplicate_rejections,
                dominated_rejections=self._dominated_rejections,
                uncertified_rejections=self._uncertified_rejections,
                removed_dominated_cuts=self._removed_dominated_cuts,
                propagation_calls=self._propagation_calls,
                cuts_examined=self._cuts_examined,
                unit_fixings=self._unit_fixings,
                conflicts=self._conflicts,
            )

    def _masks(
        self,
        forced_one: Iterable[int],
        forced_zero: Iterable[int],
    ) -> tuple[int, int]:
        one = indices_to_mask(forced_one, self.dimension)
        zero = indices_to_mask(forced_zero, self.dimension)
        if one & zero:
            raise ValueError("forced_one and forced_zero must be disjoint")
        return one, zero

    @staticmethod
    def _evidence_is_safe(
        lower_bound: Optional[float],
        upper_bound: Optional[float],
        safety_margin: float,
    ) -> bool:
        return (
            lower_bound is not None
            and upper_bound is not None
            and not math.isnan(lower_bound)
            and math.isfinite(upper_bound)
            and lower_bound > upper_bound + safety_margin
        )

    def add(
        self,
        forced_one: Iterable[int] = (),
        forced_zero: Iterable[int] = (),
        *,
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
        safety_margin: float = 0.0,
        source: str = "safe_screening",
        certified: bool = False,
    ) -> CutInsertionResult:
        """Insert a certified forbidden pattern after dominance filtering.

        Setting ``certified=True`` is an explicit assertion by the caller that
        an external proof makes the cut safe.  Otherwise, this method accepts
        the cut only when the supplied numerical evidence satisfies the strict
        inequality ``lower_bound > upper_bound + safety_margin``.
        """
        one, zero = self._masks(forced_one, forced_zero)
        lower = finite_or_none(lower_bound, "lower_bound")
        upper = finite_or_none(upper_bound, "upper_bound")
        margin = float(safety_margin)
        if not math.isfinite(margin) or margin < 0.0:
            raise ValueError("safety_margin must be finite and nonnegative")
        safe = bool(certified) or self._evidence_is_safe(lower, upper, margin)

        with self._lock:
            self._insertion_attempts += 1
            if not safe:
                self._uncertified_rejections += 1
                return CutInsertionResult(False, "uncertified")

            exact_id = self._patterns.get((one, zero))
            if exact_id is not None:
                self._duplicate_rejections += 1
                existing = self._cuts[exact_id].view
                return CutInsertionResult(
                    False,
                    "duplicate",
                    dominating_cut=existing,
                )

            # An existing pattern with a subset of both literal sets already
            # excludes everything the proposed cut would exclude.
            for stored in self._cuts.values():
                if _subsumes(stored, one, zero):
                    self._dominated_rejections += 1
                    return CutInsertionResult(
                        False,
                        "dominated",
                        dominating_cut=stored.view,
                    )

            dominated_ids = tuple(
                cut_id
                for cut_id, stored in self._cuts.items()
                if not (one & ~stored.one_mask)
                and not (zero & ~stored.zero_mask)
            )
            for cut_id in dominated_ids:
                removed = self._cuts.pop(cut_id)
                self._patterns.pop((removed.one_mask, removed.zero_mask))

            cut_id = self._next_id
            self._next_id += 1
            view = MultiSelectorCut(
                cut_id=cut_id,
                forced_one=mask_to_indices(one),
                forced_zero=mask_to_indices(zero),
                lower_bound=lower,
                upper_bound=upper,
                safety_margin=margin,
                source=str(source),
            )
            stored = _StoredCut(one, zero, view)
            self._cuts[cut_id] = stored
            self._patterns[(one, zero)] = cut_id
            self._accepted_cuts += 1
            self._removed_dominated_cuts += len(dominated_ids)
            return CutInsertionResult(
                True,
                "accepted",
                cut=view,
                removed_cut_ids=dominated_ids,
            )

    def add_masks(
        self,
        forced_one_mask: int,
        forced_zero_mask: int,
        **kwargs: Any,
    ) -> CutInsertionResult:
        """Mask-based equivalent of :meth:`add` for a hot BnB loop."""
        one = validate_mask(forced_one_mask, self.dimension, "forced_one_mask")
        zero = validate_mask(
            forced_zero_mask,
            self.dimension,
            "forced_zero_mask",
        )
        if one & zero:
            raise ValueError("forced-one and forced-zero masks must be disjoint")
        return self.add(mask_to_indices(one), mask_to_indices(zero), **kwargs)

    def add_node_cut(
        self,
        fixings: BinaryFixings,
        *,
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
        safety_margin: float = 0.0,
        source: str = "fenchel_node_bound",
        certified: bool = False,
    ) -> CutInsertionResult:
        """Forbid a node's complete root-to-node fixing pattern.

        Using this method rather than passing only the latest branch literal
        prevents an easy but serious mistake: a node lower bound certifies a
        cut only for the conjunction of *all* root and ancestor fixings used
        to compute that bound.
        """
        if fixings.dimension != self.dimension:
            raise ValueError("node and cut-pool dimensions differ")
        return self.add_masks(
            fixings.fixed_one_mask,
            fixings.fixed_zero_mask,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            safety_margin=safety_margin,
            source=source,
            certified=certified,
        )

    def add_screening_cut(self, cut: Any) -> CutInsertionResult:
        """Insert a valid ``sksfolio.screening.ScreeningCut`` by duck typing."""
        valid = bool(getattr(cut, "valid", False))
        # Respect the oracle's explicit verdict.  In particular, do not infer
        # validity from rounded fields if its own conservative test rejected
        # the cut.
        if not valid:
            return self.add(
                getattr(cut, "forced_one", ()),
                getattr(cut, "forced_zero", ()),
                source="fenchel_screening",
                certified=False,
            )
        return self.add(
            getattr(cut, "forced_one", ()),
            getattr(cut, "forced_zero", ()),
            lower_bound=getattr(cut, "lower_bound", None),
            upper_bound=getattr(cut, "upper_bound", None),
            safety_margin=getattr(cut, "safety_margin", 0.0),
            source="fenchel_screening",
            certified=valid,
        )

    def get(self, cut_id: int) -> Optional[MultiSelectorCut]:
        """Return an active cut by ID, or ``None`` if it was never active."""
        identifier = integer_index(cut_id)
        with self._lock:
            stored = self._cuts.get(identifier)
            return None if stored is None else stored.view

    def first_conflict(
        self,
        fixings: BinaryFixings,
    ) -> Optional[MultiSelectorCut]:
        """Return the first cut directly violated by a node, without propagation."""
        if fixings.dimension != self.dimension:
            raise ValueError("node and cut-pool dimensions differ")
        with self._lock:
            snapshot = tuple(self._cuts.values())
        one = fixings.fixed_one_mask
        zero = fixings.fixed_zero_mask
        examined = 0
        conflict: Optional[MultiSelectorCut] = None
        for stored in snapshot:
            examined += 1
            if stored.one_mask & zero or stored.zero_mask & one:
                continue
            missing_one = stored.one_mask & ~one
            missing_zero = stored.zero_mask & ~zero
            if not missing_one and not missing_zero:
                conflict = stored.view
                break
        with self._lock:
            self._cuts_examined += examined
            if conflict is not None:
                self._conflicts += 1
        return conflict

    def propagate(
        self,
        fixings: BinaryFixings,
        *,
        max_rounds: Optional[int] = None,
    ) -> CutPropagationResult:
        """Apply no-good-cut unit propagation until closure or conflict.

        For a pattern with one undecided literal, choosing that literal's
        pattern value would violate the cut.  Hence a required-one literal is
        fixed to zero, while a required-zero literal is fixed to one.
        """
        if fixings.dimension != self.dimension:
            raise ValueError("node and cut-pool dimensions differ")
        if max_rounds is None:
            limit = self.dimension + 1
        else:
            limit = integer_index(max_rounds)
            if limit <= 0:
                raise ValueError("max_rounds must be positive")
        with self._lock:
            snapshot = tuple(self._cuts.values())

        initial_one = fixings.fixed_one_mask
        initial_zero = fixings.fixed_zero_mask
        one = initial_one
        zero = initial_zero
        rounds = 0
        examined = 0
        conflict: Optional[MultiSelectorCut] = None
        closed = False

        while rounds < limit:
            rounds += 1
            changed = False
            for stored in snapshot:
                examined += 1
                # A contradictory fixed literal already satisfies this cut.
                if stored.one_mask & zero or stored.zero_mask & one:
                    continue
                missing_one = stored.one_mask & ~one
                missing_zero = stored.zero_mask & ~zero
                undecided = missing_one.bit_count() + missing_zero.bit_count()
                if undecided == 0:
                    conflict = stored.view
                    break
                if undecided != 1:
                    continue
                if missing_one:
                    zero |= missing_one
                else:
                    one |= missing_zero
                changed = True
            if conflict is not None:
                break
            if not changed:
                closed = True
                break

        result_fixings = BinaryFixings(
            self.dimension,
            fixed_one_mask=one,
            fixed_zero_mask=zero,
        )
        new_one = one & ~initial_one
        new_zero = zero & ~initial_zero
        fixed_count = (new_one | new_zero).bit_count()
        with self._lock:
            self._propagation_calls += 1
            self._cuts_examined += examined
            self._unit_fixings += fixed_count
            if conflict is not None:
                self._conflicts += 1
        return CutPropagationResult(
            fixings=result_fixings,
            infeasible=conflict is not None,
            closed=closed or conflict is not None,
            newly_fixed_one_mask=new_one,
            newly_fixed_zero_mask=new_zero,
            conflict_cut=conflict,
            rounds=rounds,
            cuts_examined=examined,
        )

    def propagate_masks(
        self,
        fixed_one_mask: int,
        fixed_zero_mask: int,
        *,
        max_rounds: Optional[int] = None,
    ) -> CutPropagationResult:
        """Mask-based propagation convenience for a branch-and-bound hot loop."""
        return self.propagate(
            BinaryFixings(
                self.dimension,
                fixed_one_mask=fixed_one_mask,
                fixed_zero_mask=fixed_zero_mask,
            ),
            max_rounds=max_rounds,
        )

    def prunes(self, fixings: BinaryFixings) -> bool:
        """Return whether cut propagation proves a node infeasible."""
        return self.propagate(fixings).infeasible

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of cuts and statistics."""
        with self._lock:
            cuts = [stored.view.to_dict() for stored in self._cuts.values()]
            stats = self.stats.to_dict()
        return {
            "dimension": self.dimension,
            "cuts": cuts,
            "stats": stats,
        }


# A concise alias for callers that do not need the implementation detail in
# the class name.
NoGoodCutPool = MultiSelectorCutPool


__all__ = ["MultiSelectorCutPool", "NoGoodCutPool"]
