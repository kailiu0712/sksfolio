"""Dense/sparse constraint-operator selection tests for PDHG."""

from __future__ import annotations

import unittest

import numpy as np
from scipy import sparse

from sksfolio.relaxation import MarkowitzInstance
from sksfolio.relaxation.pdhg.solver import (
    _constraint_adjoint,
    _constraint_forward,
    _frobenius_operator_bound,
    _gram_data,
    _prepare_problem,
    _use_dense_constraint_operator,
    solve_pdhg,
)

from tests._helpers import small_instance


class PdhgConstraintOperatorTests(unittest.TestCase):
    def test_dense_moderate_rows_use_blas_operator(self) -> None:
        instance = small_instance()
        problem = _prepare_problem(instance)
        self.assertEqual(problem.constraint_operator_storage, "dense")
        self.assertIsInstance(problem.C, np.ndarray)
        self.assertTrue(problem.C.flags.f_contiguous)

        vector = np.linspace(-0.2, 0.4, instance.dimension)
        dual = np.array([0.3, -0.7])
        original = sparse.csr_matrix(instance.C)
        expected = np.asarray(original @ vector).reshape(-1)
        expected /= problem.row_norms
        np.testing.assert_allclose(
            _constraint_forward(problem, vector),
            expected,
            rtol=1e-15,
            atol=1e-15,
        )
        np.testing.assert_allclose(
            _constraint_adjoint(problem, dual),
            np.asarray(problem.C).T @ dual,
            rtol=1e-15,
            atol=1e-15,
        )

        _, _, constraint_gram = _gram_data(problem)
        np.testing.assert_allclose(
            constraint_gram,
            problem.C @ problem.C.T,
            rtol=1e-15,
            atol=1e-15,
        )
        weight = 2.5
        expected_bound = np.sqrt(
            problem.factor_operator_scale**2
            * np.linalg.norm(problem.B) ** 2
            + weight * np.linalg.norm(problem.C) ** 2
        )
        self.assertAlmostEqual(
            _frobenius_operator_bound(problem, weight),
            expected_bound,
            places=14,
        )

    def test_genuinely_sparse_rows_remain_sparse(self) -> None:
        base = small_instance()
        dimension = base.dimension
        identity = sparse.eye(dimension, format="csr")
        instance = MarkowitzInstance(
            factor_loadings=base.B,
            expected_returns=base.mu,
            constraint_matrix=identity,
            lower_bounds=np.zeros(dimension),
            upper_bounds=np.ones(dimension),
            feasible_anchor=base.anchor,
            constraint_names=[
                f"asset_{index}" for index in range(dimension)
            ],
            k=base.k,
            perspective_weight=base.perspective_weight,
            return_reward=base.return_reward,
        )
        problem = _prepare_problem(instance)
        self.assertEqual(problem.constraint_operator_storage, "sparse")
        self.assertTrue(sparse.isspmatrix_csr(problem.C))
        self.assertAlmostEqual(
            problem.constraint_operator_density,
            1.0 / dimension,
        )

        vector = np.linspace(0.0, 1.0, dimension)
        np.testing.assert_allclose(
            _constraint_forward(problem, vector),
            vector,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            _constraint_adjoint(problem, vector),
            vector,
            rtol=0.0,
            atol=0.0,
        )

    def test_large_dense_operator_is_not_materialized(self) -> None:
        self.assertFalse(
            _use_dense_constraint_operator(
                2_001,
                1_000,
                2_001_000,
            )
        )

    def test_result_reports_selected_storage(self) -> None:
        result = solve_pdhg(
            small_instance(),
            variant="linesearch-restart",
            options={
                "threads": 1,
                "max_iterations": 20,
                "check_interval": 10,
                "min_epoch": 10,
                "max_epoch": 20,
            },
        )
        self.assertEqual(result["constraint_operator_storage"], "dense")
        self.assertEqual(result["constraint_operator_entries"], 48)
        self.assertEqual(result["constraint_operator_nnz"], 36)
        self.assertAlmostEqual(
            result["constraint_operator_density"],
            0.75,
        )


if __name__ == "__main__":
    unittest.main()
