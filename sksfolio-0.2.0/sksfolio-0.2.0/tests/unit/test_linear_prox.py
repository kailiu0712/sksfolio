"""Tests for the interval-constrained perspective proximal oracle."""

from __future__ import annotations

import unittest

import numpy as np
from scipy import sparse

from sksfolio.relaxation.fista import (
    LinearConstraintProx,
    prox_budget_details,
)
from sksfolio.relaxation.pdhg.pava import prox as pava_prox


class LinearConstraintProxTests(unittest.TestCase):
    def test_zero_rows_dispatches_to_plain_pava(self) -> None:
        rng = np.random.default_rng(711)
        values = rng.normal(size=30)
        oracle = LinearConstraintProx(
            sparse.csr_matrix((0, values.size)),
            np.empty(0),
            np.empty(0),
            7,
        )
        result = oracle.solve(values, 0.8)
        np.testing.assert_allclose(
            result.x,
            pava_prox(values, 0.8, 7),
            atol=1e-13,
        )
        self.assertEqual(result.method, "unconstrained_pava")

    def test_budget_dispatch_warm_starts_scalar_multiplier(self) -> None:
        rng = np.random.default_rng(921)
        dimension = 200
        values = rng.normal(size=dimension)
        oracle = LinearConstraintProx(
            sparse.csr_matrix(np.ones((1, dimension))),
            np.array([1.0]),
            np.array([1.0]),
            12,
        )
        first = oracle.solve(values, 0.6)
        second_values = values + 1e-4 * rng.normal(size=dimension)
        second = oracle.solve(second_values, 0.6)
        cold = prox_budget_details(second_values, 0.6, 12)
        np.testing.assert_allclose(second.x, cold.x, atol=2e-10)
        self.assertEqual(first.method, "exact_budget_scalar_brent")
        self.assertFalse(first.warm_started)
        self.assertTrue(second.warm_started)
        self.assertLessEqual(second.pava_calls, cold.evaluations)

    def test_forced_general_path_matches_exact_budget_path(self) -> None:
        rng = np.random.default_rng(117)
        dimension = 25
        values = rng.normal(size=dimension)
        matrix = sparse.csr_matrix(np.ones((1, dimension)))
        exact = LinearConstraintProx(
            matrix,
            np.array([1.0]),
            np.array([1.0]),
            6,
            tolerance=1e-10,
            max_iterations=200,
        ).solve(values, 0.9)
        general = LinearConstraintProx(
            matrix,
            np.array([1.0]),
            np.array([1.0]),
            6,
            tolerance=1e-8,
            max_iterations=500,
            use_budget_fast_path=False,
        ).solve(values, 0.9)
        self.assertTrue(general.converged)
        np.testing.assert_allclose(general.x, exact.x, atol=2e-7)
        np.testing.assert_allclose(
            general.constraint_multiplier,
            exact.constraint_multiplier,
            atol=2e-7,
        )

    def test_positive_row_rescaling_preserves_point_and_multiplier(self) -> None:
        rng = np.random.default_rng(819)
        dimension = 18
        values = rng.normal(size=dimension)
        matrix = sparse.csr_matrix(
            np.vstack(
                (
                    np.ones(dimension),
                    np.r_[np.ones(7), np.zeros(dimension - 7)],
                )
            )
        )
        lower = np.array([1.0, 0.15])
        upper = np.array([1.0, 0.65])
        scales = np.array([3.0, 0.2])
        first = LinearConstraintProx(
            matrix,
            lower,
            upper,
            5,
            tolerance=1e-7,
            max_iterations=500,
            use_budget_fast_path=False,
        ).solve(values, 0.7)
        second = LinearConstraintProx(
            sparse.diags(scales) @ matrix,
            scales * lower,
            scales * upper,
            5,
            tolerance=1e-7,
            max_iterations=500,
            use_budget_fast_path=False,
        ).solve(values, 0.7)
        self.assertTrue(first.converged)
        self.assertTrue(second.converged)
        np.testing.assert_allclose(first.x, second.x, atol=2e-7)
        np.testing.assert_allclose(
            first.constraint_multiplier,
            scales * second.constraint_multiplier,
            atol=3e-7,
        )

    def test_operator_storage_is_selected_from_density_and_size(self) -> None:
        dense = LinearConstraintProx(
            sparse.csr_matrix(np.ones((3, 8))),
            np.zeros(3),
            np.ones(3),
            2,
            use_budget_fast_path=False,
        )
        sparse_operator = sparse.eye(
            10,
            100,
            format="csr",
            dtype=float,
        )
        sparse_oracle = LinearConstraintProx(
            sparse_operator,
            np.zeros(10),
            np.ones(10),
            2,
            use_budget_fast_path=False,
        )
        self.assertEqual(dense.operator_storage, "dense")
        self.assertEqual(sparse_oracle.operator_storage, "sparse")


if __name__ == "__main__":
    unittest.main()
