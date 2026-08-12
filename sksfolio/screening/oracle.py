"""Constraint-aware Fenchel safe screening for cardinality selectors.

For any factor and interval-row dual multipliers ``p`` and ``q``, define

``s = return_reward * mu - B @ p - C.T @ q``.

The conjugate of one long-only binary perspective coordinate is a
nonnegative score ``a_i``.  A dual lower bound for a prescribed selector
pattern is a constant minus the largest feasible sum of these scores.  Thus,
forcing selectors only changes a top-k selection; arbitrary linear portfolio
rows are fully represented by ``q`` and the interval support term.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ..incumbent.evaluation import evaluate_incumbent
from ..relaxation.certificate import SafeDualCertificate
from ..relaxation.problem import MarkowitzInstance
from .result import SafeScreeningResult, ScreeningCut


def _indices(
    values: Sequence[int],
    dimension: int,
    name: str,
) -> np.ndarray:
    array = np.unique(np.asarray(tuple(values), dtype=np.int64).reshape(-1))
    if np.any(array < 0) or np.any(array >= dimension):
        raise ValueError(f"{name} contains an out-of-range index")
    return array


def _coerce_certificate(value: Any) -> SafeDualCertificate:
    if isinstance(value, SafeDualCertificate):
        return value
    candidate = getattr(value, "dual_certificate", None)
    if isinstance(candidate, SafeDualCertificate):
        return candidate
    raw = getattr(value, "raw", value)
    if isinstance(raw, Mapping):
        return SafeDualCertificate.from_result(raw)
    raise TypeError("expected a safe dual certificate or relaxation result")


def _coerce_upper_bound(
    instance: MarkowitzInstance,
    value: Any,
    feasibility_tolerance: float,
) -> tuple[float, bool, str]:
    weights = getattr(value, "weights", None)
    selectors = getattr(value, "selectors", None)
    raw = getattr(value, "raw", value)
    if weights is None and isinstance(raw, Mapping) and raw.get("x") is not None:
        weights = raw["x"]
        selectors = raw.get("selectors")
    if weights is not None:
        diagnostics = evaluate_incumbent(
            instance,
            weights,
            selectors,
            feasibility_tolerance=feasibility_tolerance,
        )
        if not diagnostics["numerically_feasible"]:
            raise ValueError("the supplied incumbent is not feasible")
        return float(diagnostics["objective"]), True, "verified_incumbent"
    if isinstance(raw, Mapping):
        # A continuous relaxation's ``primal_upper_bound`` is not generally
        # feasible for the binary cardinality problem.  Only the incumbent
        # layer's explicit ``upper_bound`` is accepted here.
        candidate = raw.get("upper_bound")
        if candidate is None:
            raise ValueError(
                "the supplied result has no verified sparse upper bound"
            )
        result = float(candidate)
        if not math.isfinite(result):
            raise ValueError("upper bound must be finite")
        return result, False, "reported_result_bound"
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("upper bound must be finite")
    return result, False, "user_supplied_scalar"


def _interval_support(
    multiplier: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    positive = multiplier > 0.0
    negative = multiplier < 0.0
    if np.any(positive & ~np.isfinite(upper)):
        return math.inf
    if np.any(negative & ~np.isfinite(lower)):
        return math.inf
    result = 0.0
    if np.any(positive):
        result += float(multiplier[positive] @ upper[positive])
    if np.any(negative):
        result += float(multiplier[negative] @ lower[negative])
    return result


def _perspective_scores(argument: np.ndarray, weight: float) -> np.ndarray:
    result = np.zeros_like(argument)
    quadratic = (argument > 0.0) & (argument < weight)
    linear = argument >= weight
    result[quadratic] = argument[quadratic] ** 2 / (2.0 * weight)
    result[linear] = argument[linear] - 0.5 * weight
    return result


def _largest_indices(
    scores: np.ndarray,
    allowed: np.ndarray,
    count: int,
) -> np.ndarray:
    candidates = np.flatnonzero(allowed)
    selected_count = min(max(int(count), 0), candidates.size)
    if selected_count == 0:
        return np.empty(0, dtype=np.int64)
    if selected_count == candidates.size:
        return candidates
    values = scores[candidates]
    threshold = float(
        np.partition(values, candidates.size - selected_count)[
            candidates.size - selected_count
        ]
    )
    strict = candidates[values > threshold]
    tied = np.sort(candidates[values == threshold])
    return np.sort(
        np.concatenate((strict, tied[: selected_count - strict.size]))
    ).astype(np.int64, copy=False)


class FenchelScreeningOracle:
    """Reusable $O(d)$ screening oracle from one safe dual certificate."""

    def __init__(
        self,
        instance: MarkowitzInstance,
        certificate: Any,
    ) -> None:
        instance.validate()
        self.instance = instance
        self.certificate = _coerce_certificate(certificate)
        recomputed = self.certificate.recompute(instance)
        lower_bound = recomputed.get("dual_bound")
        if lower_bound is None:
            raise ValueError("certificate does not give a finite dual bound")
        self.factor_multiplier = np.asarray(
            recomputed["factor_dual"],
            dtype=float,
        ).reshape(-1)
        self.constraint_multiplier = np.asarray(
            recomputed["constraint_dual_original"],
            dtype=float,
        ).reshape(-1)
        priced = (
            float(instance.return_reward) * np.asarray(instance.mu)
            - np.asarray(instance.B @ self.factor_multiplier).reshape(-1)
        )
        if instance.rows:
            priced = priced - np.asarray(
                instance.C.T @ self.constraint_multiplier
            ).reshape(-1)
        self.priced_argument = np.asarray(priced, dtype=float)
        self.scores = _perspective_scores(
            self.priced_argument,
            float(instance.perspective_weight),
        )
        support = _interval_support(
            self.constraint_multiplier,
            np.asarray(instance.lower, dtype=float),
            np.asarray(instance.upper, dtype=float),
        )
        self.base_value = (
            -0.5 * float(self.factor_multiplier @ self.factor_multiplier)
            - support
        )
        self.global_lower_bound = self.pattern_lower_bound()
        self.certificate_lower_bound = float(lower_bound)
        self.consistency_error = abs(
            self.global_lower_bound - self.certificate_lower_bound
        )

    def _sets(
        self,
        forced_one: Sequence[int],
        forced_zero: Sequence[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        one = _indices(
            forced_one,
            self.instance.dimension,
            "forced_one",
        )
        zero = _indices(
            forced_zero,
            self.instance.dimension,
            "forced_zero",
        )
        return one, zero

    def pattern_lower_bound(
        self,
        forced_one: Sequence[int] = (),
        forced_zero: Sequence[int] = (),
    ) -> float:
        """Lower-bound every solution matching one binary pattern."""
        one, zero = self._sets(forced_one, forced_zero)
        if np.intersect1d(one, zero).size or one.size > self.instance.k:
            return math.inf
        allowed = np.ones(self.instance.dimension, dtype=bool)
        allowed[one] = False
        allowed[zero] = False
        capacity = self.instance.k - one.size
        selected = _largest_indices(self.scores, allowed, capacity)
        return float(
            self.base_value
            - float(np.sum(self.scores[one]))
            - float(np.sum(self.scores[selected]))
        )

    def _branch_bounds(
        self,
        forced_one: np.ndarray,
        forced_zero: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        dimension = self.instance.dimension
        if (
            np.intersect1d(forced_one, forced_zero).size
            or forced_one.size > self.instance.k
        ):
            return (
                math.inf,
                np.full(dimension, math.inf),
                np.full(dimension, math.inf),
                np.empty(0, dtype=np.int64),
            )
        free = np.ones(dimension, dtype=bool)
        free[forced_one] = False
        free[forced_zero] = False
        capacity = self.instance.k - forced_one.size
        selected = _largest_indices(self.scores, free, capacity)
        node_bound = float(
            self.base_value
            - float(np.sum(self.scores[forced_one]))
            - float(np.sum(self.scores[selected]))
        )
        lower_if_zero = np.full(dimension, node_bound)
        lower_if_one = np.full(dimension, node_bound)
        lower_if_zero[forced_one] = math.inf
        lower_if_one[forced_zero] = math.inf

        selected_mask = np.zeros(dimension, dtype=bool)
        selected_mask[selected] = True
        outside = free & ~selected_mask
        replacement = (
            float(np.max(self.scores[outside])) if np.any(outside) else 0.0
        )
        if selected.size:
            lower_if_zero[selected] = (
                node_bound + self.scores[selected] - replacement
            )
        if capacity <= 0:
            lower_if_one[free] = math.inf
        elif selected.size:
            threshold = float(np.min(self.scores[selected]))
            outside_indices = np.flatnonzero(outside)
            lower_if_one[outside_indices] = (
                node_bound + threshold - self.scores[outside_indices]
            )
        return node_bound, lower_if_zero, lower_if_one, selected

    @staticmethod
    def _margin(
        upper_bound: float,
        lower_bound: float,
        absolute_margin: float,
        relative_margin: float,
    ) -> float:
        scale = max(
            1.0,
            abs(float(upper_bound)),
        )
        if math.isfinite(float(lower_bound)):
            scale = max(scale, abs(float(lower_bound)))
        return float(absolute_margin) + float(relative_margin) * scale

    def no_good_cut(
        self,
        forced_one: Sequence[int],
        forced_zero: Sequence[int],
        upper_bound: float,
        *,
        absolute_margin: float = 0.0,
        relative_margin: float = 1e-10,
    ) -> ScreeningCut:
        """Generate the second paper's multi-selector cut when certified."""
        one, zero = self._sets(forced_one, forced_zero)
        if np.intersect1d(one, zero).size:
            raise ValueError("forced_one and forced_zero must be disjoint")
        lower = self.pattern_lower_bound(one, zero)
        margin = self._margin(
            upper_bound,
            lower,
            absolute_margin,
            relative_margin,
        )
        return ScreeningCut(
            forced_one=tuple(int(value) for value in one),
            forced_zero=tuple(int(value) for value in zero),
            lower_bound=lower,
            upper_bound=float(upper_bound),
            valid=bool(lower > float(upper_bound) + margin),
            safety_margin=margin,
        )

    def screen(
        self,
        upper_bound: Any,
        *,
        forced_one: Sequence[int] = (),
        forced_zero: Sequence[int] = (),
        propagate: bool = True,
        absolute_margin: float = 0.0,
        relative_margin: float = 1e-10,
        feasibility_tolerance: float = 1e-7,
    ) -> SafeScreeningResult:
        """Safely fix selectors and optionally propagate until closure."""
        upper, upper_verified, upper_source = _coerce_upper_bound(
            self.instance,
            upper_bound,
            feasibility_tolerance,
        )
        initial_one, initial_zero = self._sets(forced_one, forced_zero)
        one = initial_one.copy()
        zero = initial_zero.copy()
        history = []
        prunable = False
        lower_if_zero = np.full(self.instance.dimension, math.nan)
        lower_if_one = np.full(self.instance.dimension, math.nan)
        node_bound = math.inf
        selected = np.empty(0, dtype=np.int64)
        maximum_rounds = self.instance.dimension + 1 if propagate else 1
        for round_index in range(maximum_rounds):
            node_bound, lower_if_zero, lower_if_one, selected = (
                self._branch_bounds(one, zero)
            )
            margin = self._margin(
                upper,
                node_bound,
                absolute_margin,
                relative_margin,
            )
            if node_bound > upper + margin:
                prunable = True
                history.append(
                    {
                        "round": round_index + 1,
                        "node_lower_bound": node_bound,
                        "fixed_zero": 0,
                        "fixed_one": 0,
                        "prunable": True,
                    }
                )
                break
            fixed = np.zeros(self.instance.dimension, dtype=bool)
            fixed[one] = True
            fixed[zero] = True
            free = ~fixed
            one_scales = np.where(
                np.isfinite(lower_if_one),
                np.abs(lower_if_one),
                1.0,
            )
            zero_scales = np.where(
                np.isfinite(lower_if_zero),
                np.abs(lower_if_zero),
                1.0,
            )
            one_margins = float(absolute_margin) + float(relative_margin) * (
                np.maximum(
                    np.maximum(1.0, abs(upper)),
                    one_scales,
                )
            )
            zero_margins = float(absolute_margin) + float(relative_margin) * (
                np.maximum(
                    np.maximum(1.0, abs(upper)),
                    zero_scales,
                )
            )
            new_zero = np.flatnonzero(
                free
                & (
                    np.isposinf(lower_if_one)
                    | (lower_if_one > upper + one_margins)
                )
            )
            new_one = np.flatnonzero(
                free
                & (
                    np.isposinf(lower_if_zero)
                    | (lower_if_zero > upper + zero_margins)
                )
            )
            overlap = np.intersect1d(new_zero, new_one)
            if overlap.size:
                prunable = True
            history.append(
                {
                    "round": round_index + 1,
                    "node_lower_bound": node_bound,
                    "fixed_zero": int(new_zero.size),
                    "fixed_one": int(new_one.size),
                    "prunable": prunable,
                }
            )
            if prunable or (new_zero.size == 0 and new_one.size == 0):
                break
            zero = np.union1d(zero, new_zero)
            one = np.union1d(one, new_one)
            if not propagate:
                break

        # In one-pass mode the returned sets include the discovered fixings;
        # report branch bounds for that resulting node, not its parent.
        changed_in_one_pass = bool(
            np.setdiff1d(zero, initial_zero).size
            or np.setdiff1d(one, initial_one).size
        )
        if not prunable and not propagate and changed_in_one_pass:
            node_bound, lower_if_zero, lower_if_one, selected = (
                self._branch_bounds(one, zero)
            )

        newly_zero = np.setdiff1d(zero, initial_zero, assume_unique=True)
        newly_one = np.setdiff1d(one, initial_one, assume_unique=True)
        free = np.ones(self.instance.dimension, dtype=bool)
        free[one] = False
        free[zero] = False
        return SafeScreeningResult(
            {
                "status": "prunable" if prunable else "screened",
                "prunable": prunable,
                "fixed_zero": zero,
                "fixed_one": one,
                "newly_fixed_zero": newly_zero,
                "newly_fixed_one": newly_one,
                "remaining_free": np.flatnonzero(free),
                "screened_count": int(newly_zero.size + newly_one.size),
                "screened_fraction": float(
                    (newly_zero.size + newly_one.size)
                    / max(1, self.instance.dimension - initial_one.size - initial_zero.size)
                ),
                "node_lower_bound": node_bound,
                "global_dual_lower_bound": self.global_lower_bound,
                "certificate_lower_bound": self.certificate_lower_bound,
                "certificate_consistency_error": self.consistency_error,
                "upper_bound": upper,
                "upper_bound_verified": upper_verified,
                "upper_bound_source": upper_source,
                "lower_bound_if_zero": lower_if_zero,
                "lower_bound_if_one": lower_if_one,
                "dual_scores": self.scores.copy(),
                "priced_argument": self.priced_argument.copy(),
                "top_free_indices": selected,
                "factor_multiplier": self.factor_multiplier.copy(),
                "constraint_multiplier": self.constraint_multiplier.copy(),
                "rounds": len(history),
                "history": history,
                "absolute_margin": float(absolute_margin),
                "relative_margin": float(relative_margin),
                "safe_in_exact_arithmetic": True,
                "floating_point_certified": bool(
                    self.certificate.floating_point_certified
                    and upper_verified
                ),
                "rule": "constraint_priced_fenchel_top_k",
            }
        )


__all__ = ["FenchelScreeningOracle"]
