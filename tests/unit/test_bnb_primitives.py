"""Correctness tests for exact BnB masks, bounds, and no-good cuts."""

from __future__ import annotations

import itertools
import math
from types import SimpleNamespace
import unittest

import numpy as np

from sksfolio import FenchelScreeningOracle, SafeDualCertificate
from sksfolio.bnb import (
    BinaryFixings,
    MultiSelectorCutPool,
    indices_to_mask,
    mask_to_indices,
)
from sksfolio.bnb.bounds import ConditionalFenchelEvaluator
from sksfolio.bnb.perspective import SelectorPerspectiveOracle
from sksfolio.bnb.propagation import RowFeasibilityOracle
from sksfolio.relaxation.problem import perspective_value
from sksfolio.relaxation.pava import prox as pava_prox

from tests._helpers import small_instance


class BitMaskTests(unittest.TestCase):
    def test_masks_are_exact_above_machine_word_size(self) -> None:
        indices = (0, 1, 63, 64, 129)
        mask = indices_to_mask(indices, 130)
        self.assertEqual(mask_to_indices(mask), indices)
        fixings = BinaryFixings.from_indices(
            130,
            fixed_one=(0, 64),
            fixed_zero=(1, 129),
        )
        self.assertEqual(fixings.fixed_one, (0, 64))
        self.assertEqual(fixings.fixed_zero, (1, 129))


class CutPoolTests(unittest.TestCase):
    def test_certification_and_dominance_antichain(self) -> None:
        pool = MultiSelectorCutPool(8)
        unsafe = pool.add(
            (0, 1),
            (2,),
            lower_bound=0.9,
            upper_bound=1.0,
        )
        self.assertFalse(unsafe.accepted)
        large = pool.add((0, 1), (2,), certified=True)
        self.assertTrue(large.accepted)
        stronger = pool.add((0,), (2,), certified=True)
        self.assertTrue(stronger.accepted)
        self.assertEqual(stronger.removed_cut_ids, (large.cut.cut_id,))
        dominated = pool.add((0, 3), (2, 4), certified=True)
        self.assertFalse(dominated.accepted)
        self.assertEqual(dominated.reason, "dominated")
        self.assertEqual(len(pool), 1)

    def test_unit_propagation_reaches_closure(self) -> None:
        pool = MultiSelectorCutPool(5)
        # z0 = 1 implies z1 = 0.
        pool.add((0, 1), certified=True)
        # z1 = 0 implies z2 = 1.
        pool.add((), (1, 2), certified=True)
        # z2 = 1 implies z3 = 0.
        pool.add((2, 3), certified=True)
        result = pool.propagate(
            BinaryFixings.from_indices(5, fixed_one=(0,))
        )
        self.assertFalse(result.infeasible)
        self.assertTrue(result.closed)
        self.assertEqual(result.fixings.fixed_one, (0, 2))
        self.assertEqual(result.fixings.fixed_zero, (1, 3))

    def test_matching_forbidden_pattern_is_a_conflict(self) -> None:
        pool = MultiSelectorCutPool(4)
        pool.add((0, 2), (1,), certified=True)
        result = pool.propagate(
            BinaryFixings.from_indices(
                4,
                fixed_one=(0, 2),
                fixed_zero=(1,),
            )
        )
        self.assertTrue(result.infeasible)
        self.assertIsNotNone(result.conflict_cut)


class ConditionalBoundTests(unittest.TestCase):
    def test_fast_node_and_child_bounds_match_oracle(self) -> None:
        problem = small_instance(
            dimension=8,
            perspective_weight=0.1,
            return_reward=1.0,
        )
        certificate = SafeDualCertificate(
            lower_bound=-math.inf,
            factor_multiplier=np.linspace(-0.02, 0.03, problem.rank),
            constraint_multiplier=np.array([0.04, -0.03]),
        )
        oracle = FenchelScreeningOracle(problem, certificate)
        root_one = (0,)
        root_zero = (7,)
        evaluator = ConditionalFenchelEvaluator(
            oracle,
            indices_to_mask(root_one, problem.dimension),
            indices_to_mask(root_zero, problem.dimension),
        )
        patterns = (
            (set(root_one), set(root_zero)),
            ({0, 1}, {7}),
            ({0}, {6, 7}),
            ({0, 1, 4}, {7}),
        )
        for one, zero in patterns:
            one_mask = indices_to_mask(one, problem.dimension)
            zero_mask = indices_to_mask(zero, problem.dimension)
            expected = oracle.pattern_lower_bound(one, zero)
            self.assertAlmostEqual(
                evaluator.bound(one_mask, zero_mask),
                expected,
                places=14,
            )
            analysis = evaluator.analyze(one_mask, zero_mask)
            for position, raw_index in enumerate(analysis.free_indices):
                index = int(raw_index)
                expected_zero = oracle.pattern_lower_bound(
                    one,
                    zero | {index},
                )
                expected_one = oracle.pattern_lower_bound(
                    one | {index},
                    zero,
                )
                self.assertAlmostEqual(
                    analysis.lower_if_zero[position],
                    expected_zero,
                    places=14,
                )
                if math.isinf(expected_one):
                    self.assertTrue(
                        math.isinf(analysis.lower_if_one[position])
                    )
                else:
                    self.assertAlmostEqual(
                        analysis.lower_if_one[position],
                        expected_one,
                        places=14,
                    )


class RowFeasibilityTests(unittest.TestCase):
    def test_row_envelopes_match_binary_support_enumeration(self) -> None:
        coefficients = np.array(
            [
                [1.0, -2.0, 3.0, -4.0, 5.0, -6.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [-1.0, -1.0, 2.0, 2.0, -3.0, 4.0],
            ]
        )
        instance = SimpleNamespace(
            dimension=6,
            rows=3,
            k=3,
            C=coefficients,
            lower=np.full(3, -math.inf),
            upper=np.full(3, math.inf),
        )
        active = indices_to_mask((0, 1, 2, 3, 4), 6)
        one = indices_to_mask((0,), 6)
        zero = indices_to_mask((4, 5), 6)
        oracle = RowFeasibilityOracle(instance, active)
        result = oracle.evaluate(one, zero)

        supports = []
        for size in range(3):
            for additional in itertools.combinations(
                (1, 2, 3),
                size,
            ):
                supports.append((0,) + additional)
        expected_lower = np.full(3, math.inf)
        expected_upper = np.full(3, -math.inf)
        for support in supports:
            values = coefficients[:, support]
            expected_lower = np.minimum(
                expected_lower,
                np.sum(np.minimum(values, 0.0), axis=1),
            )
            expected_upper = np.maximum(
                expected_upper,
                np.sum(np.maximum(values, 0.0), axis=1),
            )
        np.testing.assert_allclose(result.row_lower_envelope, expected_lower)
        np.testing.assert_allclose(result.row_upper_envelope, expected_upper)

    def test_impossible_row_is_rejected(self) -> None:
        instance = SimpleNamespace(
            dimension=5,
            rows=1,
            k=2,
            C=np.ones((1, 5)),
            lower=np.array([2.5]),
            upper=np.array([math.inf]),
        )
        oracle = RowFeasibilityOracle(
            instance,
            indices_to_mask(range(5), 5),
        )
        result = oracle.evaluate(0, 0)
        self.assertFalse(result.feasible)
        self.assertAlmostEqual(result.maximum_lower_shortfall, 0.5)


class SelectorPerspectiveTests(unittest.TestCase):
    @staticmethod
    def _coordinate_score(argument: float, weight: float) -> float:
        point = min(max(argument / weight, 0.0), 1.0)
        return argument * point - 0.5 * weight * point * point

    def test_root_value_and_prox_match_existing_pava(self) -> None:
        rng = np.random.default_rng(73)
        for dimension, k in ((7, 3), (20, 6), (20, 20)):
            values = rng.normal(size=dimension)
            point = rng.uniform(0.0, 0.3, size=dimension)
            point *= min(1.0, 0.8 * k / float(np.sum(point)))
            gamma = 0.7
            oracle = SelectorPerspectiveOracle(dimension, k)
            np.testing.assert_allclose(
                oracle.prox(values, gamma),
                pava_prox(values, gamma, k, "partial_sort"),
                atol=2e-13,
            )
            self.assertAlmostEqual(
                oracle.value(point),
                perspective_value(point, k),
                places=13,
            )

    def test_node_value_decomposes_and_capacity_endpoints(self) -> None:
        point = np.array([0.2, 0.1, 0.0, 0.3, 0.0, 0.0])
        oracle = SelectorPerspectiveOracle(
            6,
            3,
            forced_one=(0, 1),
            forced_zero=(4,),
        )
        expected = 0.5 * float(point[:2] @ point[:2])
        expected += perspective_value(point[[2, 3, 5]], 1)
        self.assertAlmostEqual(oracle.value(point), expected, places=14)

        no_capacity = SelectorPerspectiveOracle(
            6,
            2,
            forced_one=(0, 1),
            forced_zero=(4,),
        )
        prox = no_capacity.prox(np.arange(1.0, 7.0), 0.5)
        np.testing.assert_allclose(prox[[2, 3, 4, 5]], 0.0)
        self.assertTrue(math.isinf(no_capacity.value(point)))

        separable = SelectorPerspectiveOracle(
            6,
            5,
            forced_one=(0,),
            forced_zero=(5,),
        )
        values = np.linspace(-1.0, 2.0, 6)
        expected_prox = np.clip(values / 1.4, 0.0, 1.0)
        expected_prox[5] = 0.0
        np.testing.assert_allclose(
            separable.prox(values, 0.4),
            expected_prox,
            atol=1e-14,
        )

    def test_conjugate_matches_every_explicit_free_support(self) -> None:
        argument = np.array([0.2, 0.8, 0.4, -0.3, 1.5, 0.1, 0.9])
        weight = 1.2
        oracle = SelectorPerspectiveOracle(
            7,
            3,
            forced_one=(0,),
            forced_zero=(6,),
        )
        free = tuple(int(index) for index in oracle.free_indices)
        fixed = self._coordinate_score(argument[0], weight)
        best = -math.inf
        for size in range(oracle.remaining_capacity + 1):
            for support in itertools.combinations(free, size):
                value = fixed + sum(
                    self._coordinate_score(argument[index], weight)
                    for index in support
                )
                best = max(best, value)
        self.assertAlmostEqual(
            oracle.conjugate(argument, weight),
            best,
            places=14,
        )

    def test_conjugate_gradient_matches_finite_differences(self) -> None:
        argument = np.array([0.2, 0.8, 0.4, -0.3, 1.5, 0.1, 0.9])
        weight = 1.2
        oracle = SelectorPerspectiveOracle(
            7,
            3,
            forced_one=(0,),
            forced_zero=(6,),
        )
        selected = oracle.selected_free_indices(argument, weight)
        expected = np.zeros(argument.size)
        expected[oracle.forced_one] = np.clip(
            argument[oracle.forced_one] / weight,
            0.0,
            1.0,
        )
        expected[selected] = np.clip(
            argument[selected] / weight,
            0.0,
            1.0,
        )
        step = 1e-6
        numerical = np.zeros(argument.size)
        for index in range(argument.size):
            direction = np.zeros(argument.size)
            direction[index] = step
            numerical[index] = (
                oracle.conjugate(argument + direction, weight)
                - oracle.conjugate(argument - direction, weight)
            ) / (2.0 * step)
        np.testing.assert_allclose(numerical, expected, atol=2e-10)


if __name__ == "__main__":
    unittest.main()
