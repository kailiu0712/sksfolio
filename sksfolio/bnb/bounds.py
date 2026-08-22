"""Fast conditional Fenchel bounds used by branch-and-bound."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Optional, Sequence

import numpy as np

from ..relaxation.certificate import SafeDualCertificate
from ..screening.oracle import FenchelScreeningOracle


def mask_from_indices(values: Iterable[int], dimension: int) -> int:
    """Pack selector indices into an immutable Python integer bit mask."""
    result = 0
    for raw in values:
        index = int(raw)
        if index < 0 or index >= int(dimension):
            raise ValueError("selector index is out of range")
        result |= 1 << index
    return result


def indices_from_mask(mask: int) -> np.ndarray:
    """Unpack a nonnegative Python integer mask in increasing order."""
    value = int(mask)
    if value < 0:
        raise ValueError("selector mask must be nonnegative")
    result = []
    while value:
        least = value & -value
        result.append(least.bit_length() - 1)
        value ^= least
    return np.asarray(result, dtype=np.int64)


def iter_mask(mask: int):
    """Yield set-bit indices without allocating an intermediate array."""
    value = int(mask)
    while value:
        least = value & -value
        yield least.bit_length() - 1
        value ^= least


@dataclass(frozen=True)
class ConditionalAnalysis:
    lower_bound: float
    free_indices: np.ndarray
    lower_if_zero: np.ndarray
    lower_if_one: np.ndarray
    selected_mask: int
    certificate_index: int = 0


class ConditionalFenchelEvaluator:
    """One certificate restricted to a safely screened root domain."""

    def __init__(
        self,
        oracle: FenchelScreeningOracle,
        root_one_mask: int,
        root_zero_mask: int,
    ) -> None:
        self.oracle = oracle
        self.dimension = int(oracle.instance.dimension)
        self.k = int(oracle.instance.k)
        self.all_mask = (1 << self.dimension) - 1
        self.root_one_mask = int(root_one_mask)
        self.root_zero_mask = int(root_zero_mask)
        if self.root_one_mask & self.root_zero_mask:
            raise ValueError("root selector fixings conflict")
        if self.root_one_mask.bit_count() > self.k:
            raise ValueError("root fixes more than k selectors to one")
        self.root_free_mask = self.all_mask & ~(
            self.root_one_mask | self.root_zero_mask
        )
        self.scores = np.asarray(oracle.scores, dtype=float)
        free = indices_from_mask(self.root_free_mask)
        order = np.lexsort((free, -self.scores[free]))
        self.free_order = free[order]
        self.root_one_score = self._score_sum(self.root_one_mask)

    def _score_sum(self, mask: int) -> float:
        return float(sum(float(self.scores[i]) for i in iter_mask(mask)))

    def _validate_node(self, one_mask: int, zero_mask: int) -> bool:
        one = int(one_mask)
        zero = int(zero_mask)
        if one & zero or one.bit_count() > self.k:
            return False
        if (one & self.root_one_mask) != self.root_one_mask:
            raise ValueError("node omits a root fixed-one selector")
        if (zero & self.root_zero_mask) != self.root_zero_mask:
            raise ValueError("node omits a root fixed-zero selector")
        return True

    def bound(self, one_mask: int, zero_mask: int) -> float:
        """Return the pattern bound without allocating dimension-sized data."""
        one = int(one_mask)
        zero = int(zero_mask)
        if not self._validate_node(one, zero):
            return math.inf
        capacity = self.k - one.bit_count()
        selected_sum = 0.0
        selected_count = 0
        fixed = one | zero
        for raw in self.free_order:
            index = int(raw)
            if fixed & (1 << index):
                continue
            if selected_count >= capacity:
                break
            selected_sum += float(self.scores[index])
            selected_count += 1
        extra_one = one & ~self.root_one_mask
        return float(
            self.oracle.base_value
            - self.root_one_score
            - self._score_sum(extra_one)
            - selected_sum
        )

    def analyze(
        self,
        one_mask: int,
        zero_mask: int,
        free_indices: Optional[np.ndarray] = None,
    ) -> ConditionalAnalysis:
        """Return the node and all one-coordinate child lower bounds."""
        one = int(one_mask)
        zero = int(zero_mask)
        if free_indices is None:
            free_indices = indices_from_mask(
                self.root_free_mask & ~(one | zero)
            )
        else:
            free_indices = np.asarray(free_indices, dtype=np.int64).reshape(-1)
        if not self._validate_node(one, zero):
            infinite = np.full(free_indices.size, math.inf)
            return ConditionalAnalysis(
                math.inf,
                free_indices,
                infinite.copy(),
                infinite.copy(),
                0,
            )

        capacity = self.k - one.bit_count()
        fixed = one | zero
        selected = []
        replacement = 0.0
        for raw in self.free_order:
            index = int(raw)
            if fixed & (1 << index):
                continue
            if len(selected) < capacity:
                selected.append(index)
            else:
                replacement = float(self.scores[index])
                break
        selected_mask = mask_from_indices(selected, self.dimension)
        selected_sum = float(np.sum(self.scores[selected])) if selected else 0.0
        extra_one = one & ~self.root_one_mask
        node_bound = float(
            self.oracle.base_value
            - self.root_one_score
            - self._score_sum(extra_one)
            - selected_sum
        )
        lower_if_zero = np.full(free_indices.size, node_bound)
        lower_if_one = np.full(free_indices.size, node_bound)
        threshold = (
            float(self.scores[selected[-1]]) if selected else 0.0
        )
        for position, raw in enumerate(free_indices):
            index = int(raw)
            bit = 1 << index
            if selected_mask & bit:
                lower_if_zero[position] = (
                    node_bound + float(self.scores[index]) - replacement
                )
            elif capacity <= 0:
                lower_if_one[position] = math.inf
            elif selected:
                lower_if_one[position] = (
                    node_bound + threshold - float(self.scores[index])
                )
        return ConditionalAnalysis(
            lower_bound=node_bound,
            free_indices=free_indices,
            lower_if_zero=lower_if_zero,
            lower_if_one=lower_if_one,
            selected_mask=selected_mask,
        )


class FenchelCertificatePool:
    """Maximize every conditional bound over complete dual certificates."""

    def __init__(
        self,
        oracle: FenchelScreeningOracle,
        root_one_mask: int,
        root_zero_mask: int,
        maximum_size: int = 8,
    ) -> None:
        if int(maximum_size) < 1:
            raise ValueError("certificate pool size must be positive")
        self.instance = oracle.instance
        self.root_one_mask = int(root_one_mask)
        self.root_zero_mask = int(root_zero_mask)
        self.maximum_size = int(maximum_size)
        self.evaluators = [
            ConditionalFenchelEvaluator(
                oracle,
                self.root_one_mask,
                self.root_zero_mask,
            )
        ]
        self.added = 1
        self.replaced = 0

    @property
    def size(self) -> int:
        return len(self.evaluators)

    @property
    def root_oracle(self) -> FenchelScreeningOracle:
        return self.evaluators[0].oracle

    def add_oracle(self, oracle: FenchelScreeningOracle) -> bool:
        for evaluator in self.evaluators:
            same_factor = np.array_equal(
                evaluator.oracle.factor_multiplier,
                oracle.factor_multiplier,
            )
            same_constraint = np.array_equal(
                evaluator.oracle.constraint_multiplier,
                oracle.constraint_multiplier,
            )
            if same_factor and same_constraint:
                return False
        candidate = ConditionalFenchelEvaluator(
            oracle,
            self.root_one_mask,
            self.root_zero_mask,
        )
        if len(self.evaluators) < self.maximum_size:
            self.evaluators.append(candidate)
        elif self.maximum_size > 1:
            # Keep the root relaxation certificate permanently and rotate
            # support-derived certificates to retain dual diversity.
            self.evaluators.pop(1)
            self.evaluators.append(candidate)
            self.replaced += 1
        else:
            return False
        self.added += 1
        return True

    def add_restricted_qp_certificate(self, result) -> bool:
        if not bool(getattr(result, "feasible", False)):
            return False
        constraint_dual = getattr(result, "constraint_dual", None)
        if constraint_dual is None:
            return False
        x = np.asarray(result.x, dtype=float).reshape(-1)
        factor = np.asarray(self.instance.B.T @ x, dtype=float).reshape(-1)
        certificate = SafeDualCertificate(
            lower_bound=-math.inf,
            factor_multiplier=factor,
            constraint_multiplier=np.asarray(
                constraint_dual,
                dtype=float,
            ).reshape(-1),
        )
        try:
            oracle = FenchelScreeningOracle(self.instance, certificate)
        except (ValueError, TypeError, FloatingPointError):
            return False
        return self.add_oracle(oracle)

    def bound(self, one_mask: int, zero_mask: int) -> float:
        return max(
            evaluator.bound(one_mask, zero_mask)
            for evaluator in self.evaluators
        )

    def analyze(self, one_mask: int, zero_mask: int) -> ConditionalAnalysis:
        free = indices_from_mask(
            self.evaluators[0].root_free_mask
            & ~(int(one_mask) | int(zero_mask))
        )
        best_bound = -math.inf
        best_selected = 0
        best_index = 0
        lower_if_zero = np.full(free.size, -math.inf)
        lower_if_one = np.full(free.size, -math.inf)
        for index, evaluator in enumerate(self.evaluators):
            analysis = evaluator.analyze(one_mask, zero_mask, free)
            lower_if_zero = np.maximum(
                lower_if_zero,
                analysis.lower_if_zero,
            )
            lower_if_one = np.maximum(
                lower_if_one,
                analysis.lower_if_one,
            )
            if analysis.lower_bound > best_bound:
                best_bound = analysis.lower_bound
                best_selected = analysis.selected_mask
                best_index = index
        return ConditionalAnalysis(
            lower_bound=float(best_bound),
            free_indices=free,
            lower_if_zero=lower_if_zero,
            lower_if_one=lower_if_one,
            selected_mask=best_selected,
            certificate_index=best_index,
        )


__all__ = [
    "ConditionalAnalysis",
    "ConditionalFenchelEvaluator",
    "FenchelCertificatePool",
    "indices_from_mask",
    "iter_mask",
    "mask_from_indices",
]
