"""Exact-enumeration tests for constrained Fenchel safe screening."""

from __future__ import annotations

import itertools
import math
import unittest

import numpy as np
from scipy import sparse

from sksfolio import (
    FenchelScreeningOracle,
    MarkowitzInstance,
    SafeDualCertificate,
    safe_screen,
    solve_relaxation,
)
from sksfolio.incumbent import solve_restricted_qp


def screening_instance() -> MarkowitzInstance:
    dimension = 8
    instance = MarkowitzInstance(
        factor_loadings=np.diag(np.linspace(0.05, 0.12, dimension)),
        expected_returns=np.array(
            [0.15, 0.12, 0.10, 0.08, 0.06, 0.04, 0.02, 0.01]
        ),
        constraint_matrix=sparse.csr_matrix(
            np.vstack(
                [
                    np.ones(dimension),
                    np.r_[np.ones(4), np.zeros(4)],
                ]
            )
        ),
        lower_bounds=np.array([1.0, 0.25]),
        upper_bounds=np.array([1.0, 0.75]),
        feasible_anchor=np.full(dimension, 1.0 / dimension),
        constraint_names=["budget", "sector"],
        k=3,
        perspective_weight=0.1,
        return_reward=1.0,
    )
    instance.validate()
    return instance


def enumerate_selector_supports(
    instance: MarkowitzInstance,
) -> list[tuple[float, frozenset[int], np.ndarray]]:
    results = []
    for size in range(1, instance.k + 1):
        for support in itertools.combinations(range(instance.dimension), size):
            solved = solve_restricted_qp(
                instance,
                support,
                solver="scipy",
                options={
                    "feasibility_tolerance": 1e-8,
                    "ftol": 1e-13,
                },
            )
            if solved.feasible:
                results.append(
                    (
                        float(solved.objective),
                        frozenset(support),
                        solved.x,
                    )
                )
    return sorted(results, key=lambda item: item[0])


def exact_pattern_value(
    enumerated: list[tuple[float, frozenset[int], np.ndarray]],
    forced_one: set[int],
    forced_zero: set[int],
) -> float:
    values = [
        objective
        for objective, support, _ in enumerated
        if forced_one.issubset(support) and support.isdisjoint(forced_zero)
    ]
    return min(values, default=math.inf)


class SafeScreeningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.problem = screening_instance()
        cls.enumerated = enumerate_selector_supports(cls.problem)
        cls.integer_upper_bound = cls.enumerated[0][0]
        cls.optimal_support = cls.enumerated[0][1]
        cls.relaxation = solve_relaxation(
            cls.problem,
            "corrected_lbfgs",
            options={
                "max_iterations": 10_000,
                "check_interval": 10,
                "tolerance": 1e-10,
            },
        )

    def test_constraint_priced_bound_matches_saved_certificate(self) -> None:
        oracle = FenchelScreeningOracle(self.problem, self.relaxation)
        self.assertAlmostEqual(
            oracle.global_lower_bound,
            self.relaxation.safe_dual_bound,
            places=12,
        )
        self.assertGreater(
            np.linalg.norm(oracle.constraint_multiplier),
            1e-6,
        )
        self.assertLessEqual(
            oracle.global_lower_bound,
            self.integer_upper_bound + 1e-10,
        )

    def test_arbitrary_dual_multipliers_bound_every_pattern(self) -> None:
        certificate = SafeDualCertificate(
            lower_bound=-math.inf,
            factor_multiplier=np.linspace(-0.02, 0.03, self.problem.rank),
            constraint_multiplier=np.array([0.04, -0.03]),
        )
        oracle = FenchelScreeningOracle(self.problem, certificate)
        patterns = [
            (set(), set()),
            ({0}, set()),
            ({7}, set()),
            (set(), {0}),
            ({0, 4}, {1, 7}),
            ({0, 1, 4}, {2, 3}),
        ]
        for forced_one, forced_zero in patterns:
            with self.subTest(one=forced_one, zero=forced_zero):
                lower = oracle.pattern_lower_bound(forced_one, forced_zero)
                exact = exact_pattern_value(
                    self.enumerated,
                    forced_one,
                    forced_zero,
                )
                self.assertLessEqual(lower, exact + 2e-7)

    def test_screened_branches_cannot_contain_the_optimum(self) -> None:
        result = safe_screen(
            self.problem,
            self.relaxation,
            self.integer_upper_bound,
        )
        self.assertGreater(result.raw["screened_count"], 0)
        self.assertFalse(result.prunable)
        for index in result.fixed_zero:
            exact = exact_pattern_value(self.enumerated, {int(index)}, set())
            self.assertGreater(exact, self.integer_upper_bound + 1e-8)
            self.assertNotIn(int(index), self.optimal_support)
        for index in result.fixed_one:
            exact = exact_pattern_value(self.enumerated, set(), {int(index)})
            self.assertGreater(exact, self.integer_upper_bound + 1e-8)
            self.assertIn(int(index), self.optimal_support)

    def test_node_propagation_and_no_good_cut_are_safe(self) -> None:
        oracle = FenchelScreeningOracle(self.problem, self.relaxation)
        node = oracle.screen(
            self.integer_upper_bound,
            forced_one=(0,),
            forced_zero=(7,),
        )
        self.assertIn(0, node.fixed_one)
        self.assertIn(7, node.fixed_zero)
        for index in node.newly_fixed_zero:
            exact = exact_pattern_value(
                self.enumerated,
                {0, int(index)},
                {7},
            )
            self.assertGreater(exact, self.integer_upper_bound + 1e-8)
        for index in node.newly_fixed_one:
            exact = exact_pattern_value(
                self.enumerated,
                {0},
                {7, int(index)},
            )
            self.assertGreater(exact, self.integer_upper_bound + 1e-8)

        cut = oracle.no_good_cut(
            forced_one=(7,),
            forced_zero=(),
            upper_bound=self.integer_upper_bound,
        )
        self.assertTrue(cut.valid)
        for objective, support, _ in self.enumerated:
            selectors = np.zeros(self.problem.dimension)
            selectors[list(support)] = 1.0
            if cut.excludes(selectors):
                self.assertGreater(objective, self.integer_upper_bound + 1e-8)

    def test_relaxation_primal_value_is_not_accepted_as_incumbent(self) -> None:
        oracle = FenchelScreeningOracle(self.problem, self.relaxation)
        with self.assertRaises(ValueError):
            oracle.screen(
                {"primal_upper_bound": self.relaxation.primal_upper_bound}
            )

    def test_conflicting_node_is_prunable(self) -> None:
        oracle = FenchelScreeningOracle(self.problem, self.relaxation)
        result = oracle.screen(
            self.integer_upper_bound,
            forced_one=(2,),
            forced_zero=(2,),
        )
        self.assertTrue(result.prunable)
        with self.assertRaises(ValueError):
            oracle.no_good_cut((2,), (2,), self.integer_upper_bound)


if __name__ == "__main__":
    unittest.main()
