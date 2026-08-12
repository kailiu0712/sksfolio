"""Problem-model and finite perspective-value tests."""

from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np
from scipy import sparse

from sksfolio.relaxation import (
    evaluate_solution,
    perspective_value,
    solve_relaxation,
)

from tests._helpers import small_instance


def _reference_value(x: np.ndarray, k: int) -> float:
    positive = x[x > 0.0]
    if positive.size <= k:
        return 0.5 * float(positive @ positive)
    lower = 1.0
    upper = 2.0
    while np.minimum(1.0, upper * positive).sum() < k:
        upper *= 2.0
    for _ in range(100):
        middle = 0.5 * (lower + upper)
        if np.minimum(1.0, middle * positive).sum() < k:
            lower = middle
        else:
            upper = middle
    z = np.minimum(1.0, upper * positive)
    return 0.5 * float(np.sum(positive * positive / z))


class PerspectiveValueTests(unittest.TestCase):
    def test_zero_linear_rows_use_plain_pava(self) -> None:
        base = small_instance()
        problem = replace(
            base,
            constraint_matrix=sparse.csr_matrix(
                (0, base.dimension),
                dtype=float,
            ),
            lower_bounds=np.empty(0),
            upper_bounds=np.empty(0),
            constraint_names=[],
        )
        problem.validate()
        diagnostics = evaluate_solution(problem, problem.anchor)
        self.assertEqual(
            diagnostics["violations"]["linear_rows"],
            0.0,
        )
        result = solve_relaxation(
            problem,
            options={
                "threads": 1,
                "tolerance": 1e-5,
                "max_iterations": 500,
            },
        )
        self.assertEqual(result.status, "converged")
        self.assertEqual(
            result.raw["prox_oracle_used"],
            "unconstrained_pava",
        )
        self.assertTrue(result.dual_certificate.verify(problem))

    def test_finite_breakpoints_match_independent_reference(self) -> None:
        rng = np.random.default_rng(611)
        for dimension in (5, 20, 100):
            for k in (1, max(1, dimension // 4), dimension):
                for _ in range(20):
                    x = rng.exponential(size=dimension)
                    x /= max(float(np.sum(x)), 1.0)
                    x *= min(1.0, float(k) / float(np.sum(x)))
                    np.minimum(x, 1.0, out=x)
                    self.assertAlmostEqual(
                        perspective_value(x, k),
                        _reference_value(x, k),
                        places=12,
                    )

    def test_domain_is_long_only_box_with_budget(self) -> None:
        self.assertTrue(
            np.isinf(perspective_value(np.array([-0.1, 0.2]), 1))
        )
        self.assertTrue(
            np.isinf(perspective_value(np.array([1.1, 0.0]), 1))
        )
        self.assertTrue(
            np.isinf(perspective_value(np.array([0.6, 0.6]), 1))
        )
        with self.assertRaisesRegex(ValueError, "k must"):
            perspective_value(np.ones(2), 0)


if __name__ == "__main__":
    unittest.main()
