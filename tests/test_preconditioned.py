"""Check that column preconditioning preserves the portfolio model."""
import unittest

import numpy as np

from sparsefolio import make_problem, solve, warmup
from sparsefolio.core import evaluate
from sparsefolio.improvements.alm_preconditioned import transform_problem


class ColumnEquilibrationChecks(unittest.TestCase):
    def test_primal_and_dual_values_are_invariant(self):
        for n, k in ((50, 5), (100, 20)):
            with self.subTest(n=n, k=k):
                p = make_problem(n=n, k=k, exposures=15, seed=0)
                scaled, diagonal = transform_problem(p)
                x = p.anchor.copy()
                w = x / diagonal
                y = np.zeros(len(p.lo))
                y[0] = .1  # The unscaled budget multiplier carries over.
                np.testing.assert_allclose(scaled.B @ w, p.B @ x, atol=1e-13)
                np.testing.assert_allclose(scaled.K @ w, p.K @ x, atol=1e-13)
                np.testing.assert_allclose(scaled.raw_K @ w, p.raw_K @ x, atol=1e-13)
                before = evaluate(p, x, y)
                after = evaluate(scaled, w, y)
                for name in ("objective", "lower_bound", "dual_gap",
                             "rel_gap", "feasibility", "z_sum"):
                    self.assertAlmostEqual(before[name], after[name], places=10,
                                           msg=name)

    def test_public_solver_returns_original_coordinate_certificate(self):
        warmup()
        p = make_problem(n=100, k=10, exposures=10, seed=0)
        result = solve(p, method="alm_preconditioned", time_limit=5)
        self.assertTrue(result["solved"], result["metrics"])
        self.assertEqual(result["diagnostics"]["mode"], "alm_preconditioned")
        self.assertGreater(result["diagnostics"]["preprocessing_seconds"], 0)
        independently_checked = evaluate(p, result["x"], result["y"])
        self.assertAlmostEqual(result["metrics"]["rel_gap"],
                               independently_checked["rel_gap"], places=12)


if __name__ == "__main__":
    unittest.main()
