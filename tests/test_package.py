"""Portable checks for JIT kernels, packaging, and the public solver API."""
import importlib
import unittest
import numpy as np
import sparsefolio
from sparsefolio import core
from sparsefolio.improvements.fast_core import prox as fast_prox


class PackageChecks(unittest.TestCase):
    def test_numba_and_prox_agreement(self):
        self.assertTrue(sparsefolio.warmup()["jit_enabled"])
        rng = np.random.default_rng(8123)
        for _ in range(40):
            v = rng.normal(size=30)
            gamma = np.exp(rng.uniform(-3, 3, 30))
            upper = rng.uniform(.05, 1.0, 30)
            tau = float(np.exp(rng.uniform(-4, 2)))
            a = core.prox(v, tau, gamma, upper, 5)
            b = fast_prox(v, tau, gamma, upper, 5)
            np.testing.assert_allclose(a[0], b[0], atol=1e-9, rtol=1e-9)

    def test_guarded_value_bracket(self):
        value, z = core.value_and_z(np.array([.5, .5]), np.array([1., 9.]), np.ones(2), 1)
        self.assertAlmostEqual(value, 2.5, places=11)
        np.testing.assert_allclose(z, [.5, .5], atol=1e-12)
        value, z = core.value_and_z(np.array([.4, .4, .05]), np.array([1., 100., 1.]), np.ones(3), 1)
        np.testing.assert_allclose(z, [.4, .55, .05], atol=1e-11)
        self.assertAlmostEqual(value, .5*(.4+16/.55+.05), places=9)
        value, _ = core.value_and_z(np.array([.6, .6]), np.ones(2), np.ones(2), 1)
        self.assertTrue(np.isinf(value))

    def test_all_modules_import_without_mosek(self):
        for name in ("first_order", "newton", "alm_newton", "reference",
                     "improvements.alm_fast", "improvements.alm_polished",
                     "improvements.active_set", "improvements.nested_fast",
                     "improvements.inner_dual_fast", "improvements.ipm"):
            importlib.import_module("sparsefolio."+name)
        self.assertEqual(len(sparsefolio.methods()), 16)
        self.assertEqual(len(sparsefolio.methods(include_mosek=True)), 18)

    def test_recommended_solver_certificate(self):
        sparsefolio.warmup()
        p = sparsefolio.make_problem(n=100, k=10, exposures=10, seed=0)
        result = sparsefolio.solve(p, time_limit=5)
        self.assertTrue(result["solved"], result["metrics"])
        self.assertEqual(p.K.shape, (12, 100))

    def test_structured_inner_directions(self):
        from sparsefolio.improvements.inner_dual_fast import run_checks
        report = run_checks()
        self.assertEqual(report["cases"], 80)
        self.assertLess(report["max_relative_linear_residual"], 2e-8)


if __name__ == "__main__":
    unittest.main()
