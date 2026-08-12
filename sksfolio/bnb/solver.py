"""Certificate-driven branch-and-bound for constrained sparse portfolios."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np

from ..incumbent import (
    DEFAULT_RESTRICTED_SOLVER,
    evaluate_incumbent,
    solve_incumbent,
    solve_restricted_qp,
)
from ..incumbent.support import validate_branch_indices
from ..relaxation import MarkowitzInstance, load_instance_bundle, solve_relaxation
from ..screening import FenchelScreeningOracle
from .bounds import (
    ConditionalAnalysis,
    FenchelCertificatePool,
    indices_from_mask,
    iter_mask,
    mask_from_indices,
)
from .cuts import MultiSelectorCutPool
from .node_dual import ConditionalDualPolisher
from .propagation import RowFeasibilityOracle
from .result import BranchAndBoundResult
from .types import BinaryFixings


ProblemLike = Union[MarkowitzInstance, str, Path]


@dataclass(frozen=True)
class _Node:
    node_id: int
    parent_id: Optional[int]
    one_mask: int
    zero_mask: int
    depth: int
    lower_bound: float
    branch_asset: Optional[int] = None
    branch_value: Optional[int] = None


@dataclass(frozen=True)
class _Propagation:
    one_mask: int
    zero_mask: int
    analysis: ConditionalAnalysis
    pruned: bool
    reason: str
    screening_fixings: int = 0
    cardinality_fixings: int = 0
    cut_fixings: int = 0


def _problem(value: ProblemLike) -> MarkowitzInstance:
    if isinstance(value, MarkowitzInstance):
        value.validate()
        return value
    return load_instance_bundle(value)


def _margin(
    upper: float,
    lower: float,
    absolute: float,
    relative: float,
) -> float:
    scale = max(1.0, abs(float(upper)))
    if math.isfinite(float(lower)):
        scale = max(scale, abs(float(lower)))
    return float(absolute) + float(relative) * scale


def _remaining_time(start: float, limit: Optional[float]) -> Optional[float]:
    if limit is None:
        return None
    return max(0.0, float(limit) - (time.perf_counter() - start))


def _incumbent_payload(
    instance: MarkowitzInstance,
    value: Any,
    required: np.ndarray,
    forbidden: np.ndarray,
    tolerance: float,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    if value is None:
        return None, None, math.inf
    weights = getattr(value, "weights", None)
    selectors = getattr(value, "selectors", None)
    raw = getattr(value, "raw", value)
    if isinstance(raw, Mapping):
        if weights is None:
            weights = raw.get("x")
        if selectors is None:
            selectors = raw.get("selectors")
    if weights is None:
        raise ValueError("an initial BnB incumbent must include its weights")
    diagnostics = evaluate_incumbent(
        instance,
        weights,
        selectors,
        feasibility_tolerance=tolerance,
        required_assets=required,
        forbidden_assets=forbidden,
    )
    if not diagnostics["numerically_feasible"]:
        raise ValueError("the initial BnB incumbent is not sparse-feasible")
    return (
        np.asarray(weights, dtype=float).reshape(-1).copy(),
        np.asarray(diagnostics["selectors"], dtype=float).reshape(-1).copy(),
        float(diagnostics["objective"]),
    )


class _Search:
    def __init__(
        self,
        instance: MarkowitzInstance,
        relaxation: Any,
        incumbent: Any,
        required: np.ndarray,
        forbidden: np.ndarray,
        settings: Mapping[str, Any],
        start: float,
    ) -> None:
        self.instance = instance
        self.relaxation = relaxation
        self.settings = dict(settings)
        self.start = start
        self.dimension = int(instance.dimension)
        self.k = int(instance.k)
        self.all_mask = (1 << self.dimension) - 1
        self.user_one_mask = mask_from_indices(required, self.dimension)
        self.user_zero_mask = mask_from_indices(forbidden, self.dimension)
        self.absolute_margin = float(self.settings["absolute_safety_margin"])
        self.relative_margin = float(self.settings["relative_safety_margin"])
        self.feasibility_tolerance = float(
            self.settings["feasibility_tolerance"]
        )
        self.best_x, self.best_selectors, self.upper = _incumbent_payload(
            instance,
            incumbent,
            required,
            forbidden,
            self.feasibility_tolerance,
        )
        self.best_source = (
            None if self.best_x is None else "initial_incumbent"
        )
        self.incumbent_history: list[dict[str, Any]] = []
        if self.best_x is not None:
            self.incumbent_history.append(
                {
                    "time": time.perf_counter() - self.start,
                    "upper_bound": self.upper,
                    "source": self.best_source,
                    "node_id": None,
                }
            )

        self.oracle = FenchelScreeningOracle(instance, relaxation)
        root_one = required.copy()
        root_zero = forbidden.copy()
        self.root_screen = None
        if math.isfinite(self.upper) and bool(self.settings["safe_screening"]):
            self.root_screen = self.oracle.screen(
                self.upper,
                forced_one=root_one,
                forced_zero=root_zero,
                propagate=True,
                absolute_margin=self.absolute_margin,
                relative_margin=self.relative_margin,
                feasibility_tolerance=self.feasibility_tolerance,
            )
            root_one = self.root_screen.fixed_one
            root_zero = self.root_screen.fixed_zero
        self.root_one_mask = mask_from_indices(root_one, self.dimension)
        self.root_zero_mask = mask_from_indices(root_zero, self.dimension)
        if self.root_one_mask & self.root_zero_mask:
            raise RuntimeError("root safe screening produced conflicting fixings")
        self.root_active_mask = self.all_mask & ~self.root_zero_mask
        self.root_free_mask = self.all_mask & ~(
            self.root_one_mask | self.root_zero_mask
        )
        # Because z has no direct cost and appears in no side constraint, any
        # <=k selector vector can be padded. Searching this exact target
        # removes exponentially many duplicate representations of the same x.
        self.target_cardinality = min(
            self.k,
            self.root_active_mask.bit_count(),
        )
        if self.root_one_mask.bit_count() > self.target_cardinality:
            raise RuntimeError("root fixings exceed the padded cardinality")
        self.certificates = FenchelCertificatePool(
            self.oracle,
            self.root_one_mask,
            self.root_zero_mask,
            maximum_size=int(self.settings["certificate_pool_size"]),
        )
        self.cut_pool = MultiSelectorCutPool(self.dimension)
        self.row_oracle = RowFeasibilityOracle(
            instance,
            self.root_active_mask,
            maximum_dense_entries=int(
                self.settings["row_propagation_maximum_entries"]
            ),
        )
        self.node_dual = ConditionalDualPolisher(instance)
        self.polished_patterns: set[tuple[int, int]] = set()
        self.support_cache: dict[int, Any] = {}
        self.open_nodes: list[tuple[float, int, int, _Node]] = []
        self.next_node_id = 0
        self.nodes_created = 0
        self.nodes_processed = 0
        self.nodes_branched = 0
        self.nodes_pruned_bound = 0
        self.nodes_pruned_cut = 0
        self.nodes_pruned_rows = 0
        self.nodes_pruned_cardinality = 0
        self.leaves = 0
        self.unresolved_leaves = 0
        self.unresolved_lower_bound = math.inf
        self.screening_fixings = 0
        self.cardinality_fixings = 0
        self.cut_fixings = 0
        self.restricted_qp_solves = 0
        self.restricted_qp_cache_hits = 0
        self.restricted_qp_seconds = 0.0
        self.certificates_from_qp = 0
        self.node_dual_calls = 0
        self.node_dual_iterations = 0
        self.node_dual_evaluations = 0
        self.node_dual_seconds = 0.0
        self.node_dual_improvements = 0
        self.node_dual_failures = 0
        self.root_pair_cut_candidates = 0
        self.root_pair_cuts_added = 0
        self.dynamic_cuts_added = 0
        self.cut_shrink_bound_evaluations = 0
        self.cut_literals_removed = 0
        self.history: list[dict[str, Any]] = []

    def _strictly_dominated(self, lower: float) -> bool:
        if not math.isfinite(self.upper):
            return False
        return bool(
            lower
            > self.upper
            + _margin(
                self.upper,
                lower,
                self.absolute_margin,
                self.relative_margin,
            )
        )

    def _screen_mask(
        self,
        free: np.ndarray,
        child_bounds: np.ndarray,
    ) -> int:
        result = 0
        if (
            not bool(self.settings["safe_screening"])
            or not math.isfinite(self.upper)
        ):
            return result
        for index, lower in zip(free, child_bounds):
            if math.isinf(float(lower)) and lower > 0.0:
                result |= 1 << int(index)
                continue
            if self._strictly_dominated(float(lower)):
                result |= 1 << int(index)
        return result

    def _propagate(self, one_mask: int, zero_mask: int) -> _Propagation:
        one = int(one_mask)
        zero = int(zero_mask)
        screening_count = 0
        cardinality_count = 0
        cut_count = 0
        analysis = self.certificates.analyze(one, zero)
        for _ in range(self.dimension + 1):
            if one & zero or one.bit_count() > self.target_cardinality:
                return _Propagation(
                    one,
                    zero,
                    analysis,
                    True,
                    "cardinality",
                    screening_count,
                    cardinality_count,
                    cut_count,
                )
            free_mask = self.root_active_mask & ~(one | zero)
            free_count = free_mask.bit_count()
            needed = self.target_cardinality - one.bit_count()
            if needed < 0 or needed > free_count:
                return _Propagation(
                    one,
                    zero,
                    analysis,
                    True,
                    "cardinality",
                    screening_count,
                    cardinality_count,
                    cut_count,
                )
            changed = False
            if needed == 0 and free_mask:
                zero |= free_mask
                cardinality_count += free_count
                changed = True
            elif needed == free_count and free_mask:
                one |= free_mask
                cardinality_count += free_count
                changed = True

            # Cuts are stored relative to the shared root-screened domain.
            # This avoids copying potentially millions of identical root
            # literals into every cut while preserving the exact logic.
            propagated = self.cut_pool.propagate_masks(
                one & self.root_free_mask,
                zero & self.root_free_mask,
            )
            if propagated.infeasible:
                return _Propagation(
                    one,
                    zero,
                    analysis,
                    True,
                    "cut",
                    screening_count,
                    cardinality_count,
                    cut_count,
                )
            if propagated.fixed_count:
                one = (
                    self.root_one_mask
                    | propagated.fixings.fixed_one_mask
                )
                zero = (
                    self.root_zero_mask
                    | propagated.fixings.fixed_zero_mask
                )
                cut_count += propagated.fixed_count
                changed = True

            analysis = self.certificates.analyze(one, zero)
            if self._strictly_dominated(analysis.lower_bound):
                return _Propagation(
                    one,
                    zero,
                    analysis,
                    True,
                    "bound",
                    screening_count,
                    cardinality_count,
                    cut_count,
                )
            new_zero = self._screen_mask(
                analysis.free_indices,
                analysis.lower_if_one,
            )
            new_one = self._screen_mask(
                analysis.free_indices,
                analysis.lower_if_zero,
            )
            if new_one & new_zero:
                return _Propagation(
                    one,
                    zero,
                    analysis,
                    True,
                    "bound",
                    screening_count,
                    cardinality_count,
                    cut_count,
                )
            new_one &= ~(one | zero)
            new_zero &= ~(one | zero)
            if new_one or new_zero:
                one |= new_one
                zero |= new_zero
                screening_count += (new_one | new_zero).bit_count()
                changed = True

            row_result = self.row_oracle.evaluate(
                one,
                zero,
                tolerance=float(self.settings["row_propagation_tolerance"]),
            )
            if not row_result.feasible:
                return _Propagation(
                    one,
                    zero,
                    analysis,
                    True,
                    "rows",
                    screening_count,
                    cardinality_count,
                    cut_count,
                )
            if not changed:
                return _Propagation(
                    one,
                    zero,
                    analysis,
                    False,
                    "open",
                    screening_count,
                    cardinality_count,
                    cut_count,
                )
        raise RuntimeError("BnB propagation failed to reach closure")

    def _cut_matches_support(self, support_mask: int) -> bool:
        complete = BinaryFixings(
            self.dimension,
            fixed_one_mask=support_mask & self.root_free_mask,
            fixed_zero_mask=(self.all_mask & ~support_mask)
            & self.root_free_mask,
        )
        return self.cut_pool.first_conflict(complete) is not None

    def _evaluate_support(
        self,
        support_mask: int,
        *,
        source: str,
        node_id: Optional[int],
        leaf: bool = False,
    ) -> tuple[bool, bool]:
        support = int(support_mask)
        if support.bit_count() > self.k:
            return False, True
        if self._cut_matches_support(support):
            return False, True
        cached = self.support_cache.get(support)
        if cached is None:
            remaining = _remaining_time(
                self.start,
                self.settings["time_limit"],
            )
            qp_options = dict(self.settings["restricted_qp_options"])
            if remaining is not None:
                if remaining <= 0.0:
                    return False, False
                requested_limit = qp_options.get("time_limit")
                qp_options["time_limit"] = (
                    remaining
                    if requested_limit is None
                    else min(float(requested_limit), remaining)
                )
            before = time.perf_counter()
            cached = solve_restricted_qp(
                self.instance,
                indices_from_mask(support),
                solver=str(self.settings["restricted_solver"]),
                warm_start=self.best_x,
                options=qp_options,
            )
            self.restricted_qp_seconds += time.perf_counter() - before
            self.restricted_qp_solves += 1
            if (
                bool(cached.feasible and cached.optimality_certified)
                or bool(cached.infeasibility_certified)
            ):
                self.support_cache[support] = cached
        else:
            self.restricted_qp_cache_hits += 1
        if cached.feasible:
            if self.certificates.add_restricted_qp_certificate(cached):
                self.certificates_from_qp += 1
            objective = float(cached.objective)
            if objective < self.upper:
                selectors = np.zeros(self.dimension)
                selectors[indices_from_mask(support)] = 1.0
                diagnostics = evaluate_incumbent(
                    self.instance,
                    cached.x,
                    selectors,
                    feasibility_tolerance=self.feasibility_tolerance,
                    required_assets=indices_from_mask(self.user_one_mask),
                    forbidden_assets=indices_from_mask(self.user_zero_mask),
                )
                if diagnostics["numerically_feasible"]:
                    self.best_x = np.asarray(cached.x, dtype=float).copy()
                    self.best_selectors = selectors
                    self.upper = float(diagnostics["objective"])
                    self.best_source = source
                    self.incumbent_history.append(
                        {
                            "time": time.perf_counter() - self.start,
                            "upper_bound": self.upper,
                            "source": source,
                            "node_id": node_id,
                        }
                    )
            resolved = bool(cached.optimality_certified)
            if leaf and not resolved:
                self.unresolved_leaves += 1
            return True, resolved
        explicit_infeasible = bool(cached.infeasibility_certified)
        if leaf and not explicit_infeasible:
            self.unresolved_leaves += 1
            return False, False
        return False, explicit_infeasible

    def _candidate_support(
        self,
        one_mask: int,
        analysis: ConditionalAnalysis,
    ) -> int:
        support = int(one_mask) | int(analysis.selected_mask)
        if support.bit_count() < self.target_cardinality:
            for index in analysis.free_indices:
                support |= 1 << int(index)
                if support.bit_count() >= self.target_cardinality:
                    break
        return support

    def _candidate_cut_indices(self, free: np.ndarray) -> np.ndarray:
        limit = int(self.settings["cut_candidate_limit"])
        if free.size <= limit:
            return free
        scores = self.oracle.scores[free]
        order = np.lexsort((free, -scores))
        third = max(1, limit // 3)
        capacity = min(
            max(self.target_cardinality - self.root_one_mask.bit_count(), 0),
            free.size,
        )
        boundary_start = max(0, capacity - third // 2)
        boundary = order[boundary_start : boundary_start + third]
        positions = np.unique(
            np.concatenate((order[:third], order[-third:], boundary))
        )[:limit]
        return free[positions]

    def generate_root_pair_cuts(self) -> None:
        if (
            not bool(self.settings["multi_selector_cuts"])
            or not bool(self.settings["root_pair_cuts"])
            or not math.isfinite(self.upper)
        ):
            return
        free = indices_from_mask(
            self.root_active_mask
            & ~(self.root_one_mask | self.root_zero_mask)
        )
        candidates = self._candidate_cut_indices(free)
        maximum_literals = int(self.settings["maximum_cut_literals"])
        proposals = []
        for first_position in range(candidates.size):
            first = int(candidates[first_position])
            first_bit = 1 << first
            for second_position in range(first_position + 1, candidates.size):
                second = int(candidates[second_position])
                second_bit = 1 << second
                patterns = (
                    (
                        self.root_one_mask | first_bit | second_bit,
                        self.root_zero_mask,
                    ),
                    (
                        self.root_one_mask,
                        self.root_zero_mask | first_bit | second_bit,
                    ),
                    (
                        self.root_one_mask | first_bit,
                        self.root_zero_mask | second_bit,
                    ),
                    (
                        self.root_one_mask | second_bit,
                        self.root_zero_mask | first_bit,
                    ),
                )
                for one, zero in patterns:
                    self.root_pair_cut_candidates += 1
                    extra_one = one & self.root_free_mask
                    extra_zero = zero & self.root_free_mask
                    if (
                        one.bit_count() > self.target_cardinality
                        or (extra_one | extra_zero).bit_count()
                        > maximum_literals
                    ):
                        continue
                    lower = self.certificates.bound(one, zero)
                    margin = _margin(
                        self.upper,
                        lower,
                        self.absolute_margin,
                        self.relative_margin,
                    )
                    if lower > self.upper + margin:
                        proposals.append((lower - self.upper, one, zero, lower, margin))
        proposals.sort(key=lambda item: item[0], reverse=True)
        for _, one, zero, lower, margin in proposals[
            : int(self.settings["maximum_root_cuts"])
        ]:
            inserted = self.cut_pool.add_masks(
                one & self.root_free_mask,
                zero & self.root_free_mask,
                lower_bound=lower,
                upper_bound=self.upper,
                safety_margin=margin,
                source="root_pair_fenchel",
            )
            if inserted.accepted:
                self.root_pair_cuts_added += 1

    def _learn_node_cut(self, propagation: _Propagation) -> None:
        if (
            not bool(self.settings["multi_selector_cuts"])
            or propagation.reason != "bound"
            or not math.isfinite(self.upper)
            or (
                (propagation.one_mask | propagation.zero_mask)
                & self.root_free_mask
            ).bit_count()
            > int(self.settings["maximum_cut_literals"])
        ):
            return
        extra_one = propagation.one_mask & self.root_free_mask
        extra_zero = propagation.zero_mask & self.root_free_mask
        lower = float(propagation.analysis.lower_bound)
        original_literals = (extra_one | extra_zero).bit_count()
        if (
            bool(self.settings["cut_shrinking"])
            and original_literals > 1
            and self._strictly_dominated(lower)
        ):
            extra_one, extra_zero, lower = self._shrink_cut_pattern(
                extra_one,
                extra_zero,
                lower,
            )
        removed = original_literals - (extra_one | extra_zero).bit_count()
        # A full root-to-node pattern can never be encountered again in this
        # binary tree. Store a dynamic cut only when certified shrinking made
        # it useful in another subtree.
        if removed <= 0:
            return
        margin = _margin(
            self.upper,
            lower,
            self.absolute_margin,
            self.relative_margin,
        )
        inserted = self.cut_pool.add_node_cut(
            BinaryFixings(
                self.dimension,
                extra_one,
                extra_zero,
            ),
            lower_bound=lower,
            upper_bound=self.upper,
            safety_margin=margin,
        )
        if inserted.accepted:
            self.dynamic_cuts_added += 1
            self.cut_literals_removed += removed

    def _shrink_cut_pattern(
        self,
        extra_one: int,
        extra_zero: int,
        lower_bound: float,
    ) -> tuple[int, int, float]:
        """Greedily remove literals while re-certifying the larger region."""
        one = int(extra_one)
        zero = int(extra_zero)
        lower = float(lower_bound)
        scores = np.asarray(self.oracle.scores, dtype=float)
        # Releasing a high-score fixed-one or a low-score fixed-zero literal
        # is often free in the top-k conjugate. Try those literals first.
        literals = [
            (-float(scores[index]), 0, index)
            for index in iter_mask(one)
        ]
        literals.extend(
            (float(scores[index]), 1, index)
            for index in iter_mask(zero)
        )
        literals.sort()
        limit = int(self.settings["maximum_cut_shrink_evaluations"])
        evaluations = 0
        for _, kind, index in literals:
            if evaluations >= limit:
                break
            bit = 1 << int(index)
            candidate_one = one & ~bit if kind == 0 else one
            candidate_zero = zero & ~bit if kind == 1 else zero
            candidate_lower = self.certificates.bound(
                self.root_one_mask | candidate_one,
                self.root_zero_mask | candidate_zero,
            )
            self.cut_shrink_bound_evaluations += 1
            evaluations += 1
            if self._strictly_dominated(candidate_lower):
                one = candidate_one
                zero = candidate_zero
                lower = float(candidate_lower)
        return one, zero, lower

    def _record_propagation(self, propagation: _Propagation) -> None:
        self.screening_fixings += propagation.screening_fixings
        self.cardinality_fixings += propagation.cardinality_fixings
        self.cut_fixings += propagation.cut_fixings
        if propagation.pruned:
            if propagation.reason == "bound":
                self.nodes_pruned_bound += 1
            elif propagation.reason == "cut":
                self.nodes_pruned_cut += 1
            elif propagation.reason == "rows":
                self.nodes_pruned_rows += 1
            elif propagation.reason == "cardinality":
                self.nodes_pruned_cardinality += 1

    def _polish_node_dual(
        self,
        propagation: _Propagation,
    ) -> bool:
        iterations = int(self.settings["node_dual_iterations"])
        if iterations <= 0:
            return False
        pattern = (propagation.one_mask, propagation.zero_mask)
        if pattern in self.polished_patterns:
            return False
        if (
            propagation.analysis.free_indices.size
            < int(self.settings["node_dual_minimum_free"])
            or self.nodes_processed
            > int(self.settings["node_dual_node_limit"])
            or self.nodes_processed
            % int(self.settings["node_dual_frequency"])
            != 0
        ):
            return False
        remaining = _remaining_time(
            self.start,
            self.settings["time_limit"],
        )
        if remaining is not None and remaining <= 0.0:
            return False
        self.polished_patterns.add(pattern)
        initial_index = int(propagation.analysis.certificate_index)
        initial_oracle = self.certificates.evaluators[initial_index].oracle
        before = float(propagation.analysis.lower_bound)
        try:
            result = self.node_dual.solve(
                propagation.one_mask,
                propagation.zero_mask,
                initial_oracle,
                maximum_iterations=iterations,
                memory=int(self.settings["node_dual_memory"]),
                maximum_line_search=int(
                    self.settings["node_dual_maximum_line_search"]
                ),
            )
        except (ArithmeticError, RuntimeError, ValueError):
            # Polishing is optional.  The existing certificate remains a
            # valid lower bound if the numerical optimizer cannot improve it.
            self.node_dual_failures += 1
            return False
        self.node_dual_calls += 1
        self.node_dual_iterations += result.iterations
        self.node_dual_evaluations += result.function_evaluations
        self.node_dual_seconds += result.solve_seconds
        added = self.certificates.add_oracle(result.oracle)
        if result.conditional_lower_bound > before + 1e-12 * max(
            1.0,
            abs(before),
            abs(result.conditional_lower_bound),
        ):
            self.node_dual_improvements += 1
        return added

    def _push(
        self,
        one_mask: int,
        zero_mask: int,
        depth: int,
        lower_bound: float,
        parent_id: Optional[int],
        branch_asset: Optional[int] = None,
        branch_value: Optional[int] = None,
    ) -> None:
        node = _Node(
            node_id=self.next_node_id,
            parent_id=parent_id,
            one_mask=int(one_mask),
            zero_mask=int(zero_mask),
            depth=int(depth),
            lower_bound=float(lower_bound),
            branch_asset=branch_asset,
            branch_value=branch_value,
        )
        self.next_node_id += 1
        self.nodes_created += 1
        preferred = 0
        if (
            branch_asset is not None
            and branch_value is not None
            and self.best_selectors is not None
        ):
            preferred = int(
                int(round(float(self.best_selectors[branch_asset])))
                != int(branch_value)
            )
        heapq.heappush(
            self.open_nodes,
            (node.lower_bound, preferred, node.node_id, node),
        )

    def _choose_branch(self, analysis: ConditionalAnalysis) -> tuple[int, int]:
        if analysis.free_indices.size == 0:
            raise RuntimeError("cannot branch without a free selector")
        zero_gain = np.maximum(
            analysis.lower_if_zero - analysis.lower_bound,
            0.0,
        )
        one_gain = np.maximum(
            analysis.lower_if_one - analysis.lower_bound,
            0.0,
        )
        weaker = np.nan_to_num(
            np.minimum(zero_gain, one_gain),
            nan=-math.inf,
            posinf=math.inf,
        )
        stronger = np.nan_to_num(
            np.maximum(zero_gain, one_gain),
            nan=-math.inf,
            posinf=math.inf,
        )
        rule = str(self.settings["branching_rule"])
        if rule == "max":
            primary = stronger
            secondary = weaker
        elif rule == "product":
            primary = np.nan_to_num(
                (zero_gain + 1e-16) * (one_gain + 1e-16),
                nan=-math.inf,
                posinf=math.inf,
            )
            secondary = weaker
        else:
            primary = weaker
            secondary = stronger
        order = np.lexsort(
            (
                analysis.free_indices,
                -secondary,
                -primary,
            )
        )
        position = int(order[0])
        asset = int(analysis.free_indices[position])
        if self.best_selectors is None:
            preferred_value = int(
                bool(analysis.selected_mask & (1 << asset))
            )
        else:
            preferred_value = int(
                round(float(self.best_selectors[asset]))
            )
        return asset, preferred_value

    def _global_lower_bound(self) -> float:
        candidates = [
            float(entry[0]) for entry in self.open_nodes
        ]
        if math.isfinite(self.unresolved_lower_bound):
            candidates.append(self.unresolved_lower_bound)
        # A valid incumbent satisfies U >= v*.  Hence min(U, L) remains a
        # lower bound whenever L <= v*, and it preserves the essential
        # reported invariant lower_bound <= upper_bound under roundoff or
        # when every still-open region is already dominated by the incumbent.
        if math.isfinite(self.upper):
            candidates.append(self.upper)
        if candidates:
            return min(candidates)
        return math.inf

    def _gap_reached(self) -> bool:
        if not math.isfinite(self.upper) or not self.open_nodes:
            return False
        lower = self._global_lower_bound()
        gap = max(0.0, self.upper - lower)
        target = max(
            float(self.settings["absolute_gap"]),
            float(self.settings["relative_gap"])
            * max(1.0, abs(self.upper)),
        )
        return gap <= target

    def solve(self) -> BranchAndBoundResult:
        self.generate_root_pair_cuts()
        root_bound = self.certificates.bound(
            self.root_one_mask,
            self.root_zero_mask,
        )
        self._push(
            self.root_one_mask,
            self.root_zero_mask,
            0,
            root_bound,
            None,
        )
        status = "searching"
        history_interval = int(self.settings["history_interval"])
        while self.open_nodes:
            remaining = _remaining_time(
                self.start,
                self.settings["time_limit"],
            )
            if remaining is not None and remaining <= 0.0:
                status = "time_limit"
                break
            if self.nodes_processed >= int(self.settings["node_limit"]):
                status = "node_limit"
                break
            if self._gap_reached():
                status = (
                    "optimal"
                    if self._global_lower_bound() >= self.upper
                    else "gap_limit"
                )
                break

            _, _, _, node = heapq.heappop(self.open_nodes)
            self.nodes_processed += 1
            propagation = self._propagate(node.one_mask, node.zero_mask)
            self._record_propagation(propagation)
            if propagation.pruned:
                self._learn_node_cut(propagation)
                continue
            analysis = propagation.analysis
            # A stale heap key is still safe, but reinsert a substantially
            # strengthened node so best-bound order remains meaningful.
            key_scale = max(1.0, abs(node.lower_bound), abs(analysis.lower_bound))
            if analysis.lower_bound > node.lower_bound + 1e-12 * key_scale:
                self._push(
                    propagation.one_mask,
                    propagation.zero_mask,
                    node.depth,
                    analysis.lower_bound,
                    node.parent_id,
                    node.branch_asset,
                    node.branch_value,
                )
                continue
            if self._polish_node_dual(propagation):
                polished = self._propagate(
                    propagation.one_mask,
                    propagation.zero_mask,
                )
                self._record_propagation(polished)
                if polished.pruned:
                    self._learn_node_cut(polished)
                    continue
                self._push(
                    polished.one_mask,
                    polished.zero_mask,
                    node.depth,
                    polished.analysis.lower_bound,
                    node.parent_id,
                    node.branch_asset,
                    node.branch_value,
                )
                continue
            complete = not bool(
                self.root_active_mask
                & ~(propagation.one_mask | propagation.zero_mask)
            )
            run_heuristic = bool(
                complete
                or self.nodes_processed == 1
                or self.nodes_processed
                % int(self.settings["node_heuristic_frequency"])
                == 0
            )
            if run_heuristic:
                support = self._candidate_support(
                    propagation.one_mask,
                    analysis,
                )
                _, resolved = self._evaluate_support(
                    support,
                    source=("leaf_qp" if complete else "node_score_qp"),
                    node_id=node.node_id,
                    leaf=complete,
                )
                if complete:
                    self.leaves += 1
                    if not resolved:
                        self.unresolved_lower_bound = min(
                            self.unresolved_lower_bound,
                            analysis.lower_bound,
                        )
                        status = "numerical_failure"
                        break
                    continue
                # A better incumbent or a new support certificate can close
                # the current node without branching.
                propagation = self._propagate(
                    propagation.one_mask,
                    propagation.zero_mask,
                )
                self._record_propagation(propagation)
                if propagation.pruned:
                    self._learn_node_cut(propagation)
                    continue
                analysis = propagation.analysis
                complete = not bool(
                    self.root_active_mask
                    & ~(propagation.one_mask | propagation.zero_mask)
                )
                if complete:
                    # A restricted-QP dual certificate can finish all
                    # remaining selector decisions during propagation.
                    support = self._candidate_support(
                        propagation.one_mask,
                        analysis,
                    )
                    _, resolved = self._evaluate_support(
                        support,
                        source="propagated_leaf_qp",
                        node_id=node.node_id,
                        leaf=True,
                    )
                    self.leaves += 1
                    if not resolved:
                        self.unresolved_lower_bound = min(
                            self.unresolved_lower_bound,
                            analysis.lower_bound,
                        )
                        status = "numerical_failure"
                        break
                    continue

            asset, preferred = self._choose_branch(analysis)
            bit = 1 << asset
            child_data = (
                (
                    propagation.one_mask | bit,
                    propagation.zero_mask,
                    float(
                        analysis.lower_if_one[
                            int(np.searchsorted(analysis.free_indices, asset))
                        ]
                    ),
                    1,
                ),
                (
                    propagation.one_mask,
                    propagation.zero_mask | bit,
                    float(
                        analysis.lower_if_zero[
                            int(np.searchsorted(analysis.free_indices, asset))
                        ]
                    ),
                    0,
                ),
            )
            # The heap is best-bound; insertion order is only a deterministic
            # tie breaker favoring the incumbent-compatible child.
            child_data = sorted(
                child_data,
                key=lambda child: int(child[3] != preferred),
            )
            for child_one, child_zero, child_lower, value in child_data:
                if self._strictly_dominated(child_lower):
                    self.nodes_pruned_bound += 1
                    continue
                self._push(
                    child_one,
                    child_zero,
                    node.depth + 1,
                    child_lower,
                    node.node_id,
                    asset,
                    value,
                )
            self.nodes_branched += 1
            if self.nodes_processed % history_interval == 0:
                self.history.append(
                    {
                        "time": time.perf_counter() - self.start,
                        "nodes_processed": self.nodes_processed,
                        "open_nodes": len(self.open_nodes),
                        "lower_bound": self._global_lower_bound(),
                        "upper_bound": (
                            None if not math.isfinite(self.upper) else self.upper
                        ),
                        "active_cuts": len(self.cut_pool),
                        "certificate_pool_size": self.certificates.size,
                    }
                )

        if status == "searching":
            if self.unresolved_leaves:
                status = "numerical_failure"
            elif math.isfinite(self.upper):
                status = "optimal"
            else:
                status = "infeasible"
        lower = self._global_lower_bound()
        if status == "infeasible":
            lower = math.inf
        upper_value = self.upper if math.isfinite(self.upper) else None
        lower_value = lower if math.isfinite(lower) else None
        absolute_gap = (
            None
            if upper_value is None or lower_value is None
            else max(0.0, upper_value - lower_value)
        )
        relative_gap = (
            None
            if absolute_gap is None
            else absolute_gap / max(1.0, abs(upper_value))
        )
        root_screened = (
            0 if self.root_screen is None else self.root_screen.screened_count
        )
        cut_snapshot = self.cut_pool.to_dict()
        return BranchAndBoundResult(
            {
                "status": status,
                "formulation": "safe_screened_binary_perspective_bnb",
                "x": None if self.best_x is None else self.best_x.copy(),
                "selectors": (
                    None
                    if self.best_selectors is None
                    else self.best_selectors.copy()
                ),
                "upper_bound": upper_value,
                "lower_bound": lower_value,
                "absolute_gap": absolute_gap,
                "relative_gap": relative_gap,
                "numerically_feasible": self.best_x is not None,
                "algorithmically_safe_in_exact_arithmetic": True,
                # Kept for consistency with the relaxation result schema.
                "safe_in_exact_arithmetic": True,
                "floating_point_certified": False,
                "formal_optimality_certificate": False,
                "best_incumbent_source": self.best_source,
                "nodes_created": self.nodes_created,
                "nodes_processed": self.nodes_processed,
                "nodes_branched": self.nodes_branched,
                "open_nodes": len(self.open_nodes),
                "leaves": self.leaves,
                "unresolved_leaves": self.unresolved_leaves,
                "nodes_pruned_bound": self.nodes_pruned_bound,
                "nodes_pruned_cut": self.nodes_pruned_cut,
                "nodes_pruned_rows": self.nodes_pruned_rows,
                "nodes_pruned_cardinality": self.nodes_pruned_cardinality,
                "root_fixed_one": list(iter_mask(self.root_one_mask)),
                "root_fixed_zero": list(iter_mask(self.root_zero_mask)),
                "root_screened_count": root_screened,
                "target_selector_cardinality": self.target_cardinality,
                "screening_fixings": self.screening_fixings,
                "cardinality_fixings": self.cardinality_fixings,
                "cut_fixings": self.cut_fixings,
                "cuts": cut_snapshot["cuts"],
                "cuts_are_conditional_on_root_fixings": True,
                "cut_statistics": cut_snapshot["stats"],
                "root_pair_cut_candidates": self.root_pair_cut_candidates,
                "root_pair_cuts_added": self.root_pair_cuts_added,
                "dynamic_cuts_added": self.dynamic_cuts_added,
                "cut_shrink_bound_evaluations": (
                    self.cut_shrink_bound_evaluations
                ),
                "cut_literals_removed": self.cut_literals_removed,
                "certificate_pool_size": self.certificates.size,
                "certificates_added": self.certificates.added,
                "certificates_replaced": self.certificates.replaced,
                "certificates_from_restricted_qp": self.certificates_from_qp,
                "node_dual_calls": self.node_dual_calls,
                "node_dual_iterations": self.node_dual_iterations,
                "node_dual_function_evaluations": self.node_dual_evaluations,
                "node_dual_seconds": self.node_dual_seconds,
                "node_dual_improvements": self.node_dual_improvements,
                "node_dual_failures": self.node_dual_failures,
                "restricted_qp_solves": self.restricted_qp_solves,
                "restricted_qp_cache_hits": self.restricted_qp_cache_hits,
                "restricted_qp_seconds": self.restricted_qp_seconds,
                "root_relaxation_lower_bound": self.oracle.global_lower_bound,
                "row_propagation_enabled": self.row_oracle.enabled,
                "row_propagation_storage": self.row_oracle.storage,
                "incumbent_history": self.incumbent_history,
                "history": self.history,
                "solve_seconds": time.perf_counter() - self.start,
                "settings": dict(self.settings),
            }
        )


def solve_bnb(
    problem: ProblemLike,
    *,
    relaxation: Optional[Any] = None,
    incumbent: Optional[Any] = None,
    required_assets: Sequence[int] = (),
    forbidden_assets: Sequence[int] = (),
    relaxation_backend: str = "fista",
    incumbent_method: str = "auto",
    restricted_solver: str = DEFAULT_RESTRICTED_SOLVER,
    time_limit: Optional[float] = None,
    node_limit: int = 1_000_000,
    relative_gap: float = 0.0,
    absolute_gap: float = 0.0,
    options: Optional[Mapping[str, Any]] = None,
) -> BranchAndBoundResult:
    """Solve the original binary sparse portfolio problem.

    The lower-bound engine uses complete Fenchel certificates from the
    perspective relaxation. Multi-selector cuts are strict cutoff proofs and
    are propagated as forbidden binary patterns. ``relative_gap=0`` and
    ``absolute_gap=0`` request full tree closure in exact arithmetic.
    """
    instance = _problem(problem)
    required, forbidden = validate_branch_indices(
        instance.dimension,
        instance.k,
        required_assets,
        forbidden_assets,
    )
    if time_limit is not None and (
        not math.isfinite(float(time_limit)) or float(time_limit) <= 0.0
    ):
        raise ValueError("time_limit must be positive and finite")
    if int(node_limit) < 1:
        raise ValueError("node_limit must be positive")
    for name, value in (
        ("relative_gap", float(relative_gap)),
        ("absolute_gap", float(absolute_gap)),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    supplied = dict(options or {})
    relaxation_options = dict(supplied.pop("relaxation_options", {}))
    incumbent_options = dict(supplied.pop("incumbent_options", {}))
    incumbent_time_limit = incumbent_options.pop("time_limit", None)
    for label, value in (
        ("relaxation_options.time_limit", relaxation_options.get("time_limit")),
        ("incumbent_options.time_limit", incumbent_time_limit),
        (
            "restricted_qp_options.time_limit",
            dict(supplied.get("restricted_qp_options", {})).get("time_limit"),
        ),
    ):
        if value is not None and (
            not math.isfinite(float(value)) or float(value) < 0.0
        ):
            raise ValueError(f"{label} must be finite and nonnegative")
    settings = {
        "time_limit": None if time_limit is None else float(time_limit),
        "node_limit": int(node_limit),
        "relative_gap": float(relative_gap),
        "absolute_gap": float(absolute_gap),
        "safe_screening": bool(supplied.pop("safe_screening", True)),
        "multi_selector_cuts": bool(
            supplied.pop("multi_selector_cuts", False)
        ),
        "root_pair_cuts": bool(supplied.pop("root_pair_cuts", True)),
        "branching_rule": str(
            supplied.pop("branching_rule", "max_min")
        ).lower().replace("-", "_"),
        "cut_candidate_limit": int(
            supplied.pop("cut_candidate_limit", 32)
        ),
        "maximum_root_cuts": int(supplied.pop("maximum_root_cuts", 256)),
        "maximum_cut_literals": int(
            supplied.pop("maximum_cut_literals", 256)
        ),
        "cut_shrinking": bool(supplied.pop("cut_shrinking", True)),
        "maximum_cut_shrink_evaluations": int(
            supplied.pop("maximum_cut_shrink_evaluations", 64)
        ),
        "certificate_pool_size": int(
            supplied.pop("certificate_pool_size", 8)
        ),
        "node_dual_iterations": int(
            supplied.pop("node_dual_iterations", 15)
        ),
        "node_dual_minimum_free": int(
            supplied.pop("node_dual_minimum_free", 16)
        ),
        "node_dual_node_limit": int(
            supplied.pop("node_dual_node_limit", 10_000)
        ),
        "node_dual_frequency": int(
            supplied.pop("node_dual_frequency", 1)
        ),
        "node_dual_memory": int(supplied.pop("node_dual_memory", 10)),
        "node_dual_maximum_line_search": int(
            supplied.pop("node_dual_maximum_line_search", 20)
        ),
        "node_heuristic_frequency": int(
            supplied.pop("node_heuristic_frequency", 1)
        ),
        "history_interval": int(supplied.pop("history_interval", 100)),
        "absolute_safety_margin": float(
            supplied.pop("absolute_safety_margin", 0.0)
        ),
        "relative_safety_margin": float(
            supplied.pop("relative_safety_margin", 1e-10)
        ),
        "feasibility_tolerance": float(
            supplied.pop("feasibility_tolerance", 1e-7)
        ),
        "row_propagation_tolerance": float(
            supplied.pop("row_propagation_tolerance", 1e-10)
        ),
        "row_propagation_maximum_entries": int(
            supplied.pop("row_propagation_maximum_entries", 5_000_000)
        ),
        "restricted_solver": str(restricted_solver),
        "restricted_qp_options": dict(
            supplied.pop("restricted_qp_options", {})
        ),
    }
    for name in (
        "cut_candidate_limit",
        "maximum_root_cuts",
        "maximum_cut_literals",
        "maximum_cut_shrink_evaluations",
        "certificate_pool_size",
        "node_dual_minimum_free",
        "node_dual_node_limit",
        "node_dual_frequency",
        "node_dual_memory",
        "node_dual_maximum_line_search",
        "node_heuristic_frequency",
        "history_interval",
        "row_propagation_maximum_entries",
    ):
        if int(settings[name]) < 1:
            raise ValueError(f"{name} must be positive")
    if int(settings["node_dual_iterations"]) < 0:
        raise ValueError("node_dual_iterations must be nonnegative")
    if settings["branching_rule"] not in {"max_min", "max", "product"}:
        raise ValueError("branching_rule must be max_min, max, or product")
    for name in (
        "absolute_safety_margin",
        "relative_safety_margin",
        "row_propagation_tolerance",
    ):
        value = float(settings[name])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    feasibility_tolerance = float(settings["feasibility_tolerance"])
    if (
        not math.isfinite(feasibility_tolerance)
        or feasibility_tolerance <= 0.0
    ):
        raise ValueError(
            "feasibility_tolerance must be positive and finite"
        )
    if supplied:
        unknown = ", ".join(sorted(supplied))
        raise ValueError(f"unknown BnB options: {unknown}")

    start = time.perf_counter()
    if relaxation is None:
        remaining = _remaining_time(start, time_limit)
        if remaining is not None:
            requested_limit = relaxation_options.get("time_limit")
            effective_limit = max(remaining, 1e-6)
            relaxation_options["time_limit"] = (
                effective_limit
                if requested_limit is None
                else min(float(requested_limit), effective_limit)
            )
        relaxation = solve_relaxation(
            instance,
            backend=relaxation_backend,
            options=relaxation_options,
        )
    if getattr(relaxation, "dual_certificate", None) is None:
        raise ValueError("root relaxation has no recomputable Fenchel certificate")

    if incumbent is None:
        remaining = _remaining_time(start, time_limit)
        incumbent_limit = remaining
        if incumbent_time_limit is not None:
            incumbent_limit = (
                float(incumbent_time_limit)
                if remaining is None
                else min(float(incumbent_time_limit), remaining)
            )
        if incumbent_limit is None or incumbent_limit > 0.0:
            incumbent = solve_incumbent(
                instance,
                relaxation,
                method=incumbent_method,
                restricted_solver=restricted_solver,
                required_assets=required,
                forbidden_assets=forbidden,
                time_limit=incumbent_limit,
                options=incumbent_options,
            )
            if not incumbent.feasible:
                incumbent = None
    search = _Search(
        instance,
        relaxation,
        incumbent,
        required,
        forbidden,
        settings,
        start,
    )
    return search.solve()


solve_branch_and_bound = solve_bnb


__all__ = ["solve_bnb", "solve_branch_and_bound"]
