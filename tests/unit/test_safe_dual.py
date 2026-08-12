"""Safe-dual and PDHG-variant regression tests."""

from __future__ import annotations

import unittest

from sksfolio.relaxation import evaluate_solution, solve_relaxation
from sksfolio.relaxation.pdhg import PDHG_VARIANTS, PAVA_METHODS

from tests._helpers import small_instance


class SafeDualTests(unittest.TestCase):
    def test_every_variant_and_pava_returns_recomputable_bound(self) -> None:
        problem = small_instance()
        feasible_objective = evaluate_solution(
            problem,
            problem.anchor,
        )["objective"]
        for variant in PDHG_VARIANTS:
            for pava in PAVA_METHODS:
                with self.subTest(variant=variant, pava=pava):
                    result = solve_relaxation(
                        problem,
                        "pdhg",
                        variant=variant,
                        pava=pava,
                        options={
                            "threads": 1,
                            "max_iterations": 40,
                            "check_interval": 10,
                            "min_epoch": 10,
                            "max_epoch": 20,
                        },
                    )
                    certificate = result.dual_certificate
                    self.assertIsNotNone(certificate)
                    self.assertTrue(certificate.verify(problem))
                    self.assertLessEqual(
                        result.safe_dual_bound,
                        feasible_objective + 1e-11,
                    )
                    self.assertNotIn("pava_root_solver", result)
                    self.assertNotIn("pava_uses_newton", result)
                    self.assertNotIn("pava_uses_bisection", result)
                    self.assertEqual(
                        result["maximum_dual_domain_correction"],
                        0.0,
                    )

    def test_dual_cutoff_uses_original_objective_units(self) -> None:
        problem = small_instance(perspective_weight=100.0)
        options = {
            "threads": 1,
            "max_iterations": 30,
            "check_interval": 10,
            "min_epoch": 10,
            "max_epoch": 20,
        }
        baseline = solve_relaxation(
            problem,
            "pdhg",
            variant="metric-linesearch-restart",
            options=options,
        )
        cutoff = baseline.safe_dual_bound - 1e-12
        stopped = solve_relaxation(
            problem,
            "pdhg",
            variant="metric-linesearch-restart",
            options={**options, "dual_bound_cutoff": cutoff},
        )
        self.assertEqual(stopped.status, "dual_bound_cutoff")
        self.assertGreaterEqual(stopped.safe_dual_bound, cutoff)
        self.assertEqual(stopped["dual_bound_units"], "original_objective")


if __name__ == "__main__":
    unittest.main()
