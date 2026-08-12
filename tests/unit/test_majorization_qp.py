"""Tests for the optional exact weak-majorization OSQP oracle."""

from __future__ import annotations

import importlib.util
import unittest

import numpy as np
from scipy import sparse

from sksfolio.relaxation.fista.budget_prox import prox_budget_details
from sksfolio.relaxation.fista.majorization_qp import (
    MajorizationQPOracle,
)


@unittest.skipUnless(
    importlib.util.find_spec("osqp") is not None,
    "OSQP is not installed",
)
class MajorizationQPTests(unittest.TestCase):
    def test_budget_prox_and_multiplier_match_pava_oracle(self) -> None:
        rng = np.random.default_rng(1703)
        dimension = 9
        k = 4
        argument = rng.normal(size=dimension)
        gamma = 0.7
        oracle = MajorizationQPOracle(
            sparse.csr_matrix(np.ones((1, dimension))),
            np.array([1.0]),
            np.array([1.0]),
            k,
            tolerance=1e-9,
            max_iterations=100_000,
        )
        result = oracle.solve(argument, gamma)
        expected = prox_budget_details(
            argument,
            gamma,
            k,
            tolerance=1e-11,
        )

        self.assertTrue(result.converged, result.status)
        self.assertTrue(result.has_solution)
        self.assertLess(result.maximum_constraint_violation, 2e-7)
        self.assertLess(result.majorization_violation, 2e-7)
        np.testing.assert_allclose(
            result.x,
            expected.x,
            atol=2e-6,
            rtol=2e-6,
        )
        np.testing.assert_allclose(
            result.multiplier,
            np.array([expected.eta]),
            atol=3e-6,
            rtol=3e-6,
        )

    def test_model_is_reused_with_a_warm_start(self) -> None:
        dimension = 7
        oracle = MajorizationQPOracle(
            sparse.csr_matrix((0, dimension)),
            np.empty(0),
            np.empty(0),
            k=3,
            tolerance=1e-8,
        )
        first = oracle.solve(np.linspace(-0.4, 1.1, dimension), 0.3)
        second = oracle.solve(np.linspace(-0.3, 1.0, dimension), 0.5)

        self.assertTrue(first.converged, first.status)
        self.assertFalse(first.warm_start_used)
        self.assertTrue(second.converged, second.status)
        self.assertTrue(second.warm_start_used)
        self.assertEqual(second.multiplier.shape, (0,))
        self.assertLess(second.majorization_violation, 5e-7)
        self.assertTrue(np.all(second.w[:-1] >= second.w[1:] - 5e-7))


if __name__ == "__main__":
    unittest.main()
