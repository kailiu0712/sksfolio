"""Safe-dual regression tests for both corrected algorithms."""

from __future__ import annotations

import unittest

from sksfolio.relaxation import (
    CORRECTED_ALGORITHMS,
    evaluate_solution,
    solve_relaxation,
)

from tests._helpers import small_instance


class SafeDualTests(unittest.TestCase):
    def test_bounds_are_recomputable_and_below_a_feasible_value(self) -> None:
        problem = small_instance()
        feasible_objective = evaluate_solution(problem, problem.anchor)[
            "objective"
        ]
        for algorithm in CORRECTED_ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                result = solve_relaxation(
                    problem,
                    algorithm,
                    options={
                        "threads": 1,
                        "max_iterations": 2_000,
                        "tolerance": 1e-5,
                        "prox_tolerance": 1e-7,
                    },
                )
                certificate = result.dual_certificate
                self.assertIsNotNone(certificate)
                self.assertTrue(certificate.verify(problem))
                self.assertLessEqual(
                    result.safe_dual_bound,
                    feasible_objective + 1e-11,
                )
                self.assertEqual(result["dual_bound_units"], "original_objective")
                self.assertEqual(result["maximum_dual_domain_correction"], 0.0)


if __name__ == "__main__":
    unittest.main()
