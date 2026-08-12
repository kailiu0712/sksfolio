"""Correctness tests for sparse incumbent generation."""

from __future__ import annotations

import itertools
import unittest

import numpy as np

from sksfolio.incumbent import (
    binary_perspective_prox,
    dependent_round_support,
    evaluate_incumbent,
    perspective_activations,
    solve_incumbent,
    solve_restricted_qp,
    sparse_simplex_prox,
)

from tests._helpers import small_instance


class SupportPrimitiveTests(unittest.TestCase):
    def test_binary_prox_matches_exhaustive_supports(self) -> None:
        rng = np.random.default_rng(87)
        for dimension in (4, 7):
            for k in range(1, min(3, dimension) + 1):
                for _ in range(10):
                    v = rng.normal(size=dimension)
                    step = 0.7
                    weight = 1.9
                    solution, _, _ = binary_perspective_prox(
                        v,
                        step,
                        weight,
                        k,
                    )

                    def objective(x: np.ndarray) -> float:
                        return 0.5 * float((x - v) @ (x - v)) + (
                            0.5 * step * weight * float(x @ x)
                        )

                    best = objective(np.zeros(dimension))
                    scale = 1.0 + step * weight
                    for size in range(1, k + 1):
                        for support in itertools.combinations(
                            range(dimension),
                            size,
                        ):
                            candidate = np.zeros(dimension)
                            candidate[list(support)] = np.clip(
                                v[list(support)] / scale,
                                0.0,
                                1.0,
                            )
                            best = min(best, objective(candidate))
                    self.assertAlmostEqual(objective(solution), best, places=12)

    def test_sparse_simplex_prox_matches_enumeration(self) -> None:
        rng = np.random.default_rng(911)
        dimension = 7
        k = 3
        step = 0.4
        weight = 2.0
        scale = 1.0 + step * weight
        for _ in range(12):
            v = rng.normal(size=dimension)
            solution, _ = sparse_simplex_prox(v, step, weight, k)

            def objective(x: np.ndarray) -> float:
                return 0.5 * float((x - v) @ (x - v)) + (
                    0.5 * step * weight * float(x @ x)
                )

            best = np.inf
            for size in range(1, k + 1):
                for support in itertools.combinations(range(dimension), size):
                    values = v[list(support)] / scale
                    ordered = np.sort(values)[::-1]
                    cumulative = np.cumsum(ordered) - 1.0
                    indices = np.arange(1, size + 1)
                    active = ordered - cumulative / indices > 0.0
                    rho = int(np.flatnonzero(active)[-1])
                    threshold = cumulative[rho] / float(rho + 1)
                    candidate = np.zeros(dimension)
                    candidate[list(support)] = np.maximum(
                        values - threshold,
                        0.0,
                    )
                    best = min(best, objective(candidate))
            self.assertAlmostEqual(objective(solution), best, places=11)
            self.assertAlmostEqual(float(np.sum(solution)), 1.0, places=12)
            self.assertLessEqual(np.count_nonzero(solution), k)

    def test_activation_recovery_and_dependent_rounding(self) -> None:
        rng = np.random.default_rng(773)
        x = rng.uniform(0.001, 0.2, size=50)
        x /= float(np.sum(x))
        probabilities = perspective_activations(x, 8)
        self.assertAlmostEqual(float(np.sum(probabilities)), 8.0, places=11)
        self.assertTrue(np.all(probabilities >= 0.0))
        self.assertTrue(np.all(probabilities <= 1.0))
        first = dependent_round_support(
            probabilities,
            np.random.default_rng(19),
            8,
        )
        second = dependent_round_support(
            probabilities,
            np.random.default_rng(19),
            8,
        )
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.size, 8)


class IncumbentTests(unittest.TestCase):
    def test_dense_relaxation_is_not_an_incumbent(self) -> None:
        problem = small_instance()
        diagnostics = evaluate_incumbent(problem, problem.anchor)
        self.assertFalse(diagnostics["numerically_feasible"])
        self.assertGreater(
            diagnostics["violations"]["active_cardinality"],
            0.0,
        )

    def test_restricted_qp_returns_a_verified_upper_bound(self) -> None:
        problem = small_instance()
        support = np.array([0, 1, 2, 12, 13, 14])
        result = solve_restricted_qp(
            problem,
            support,
            solver="scipy",
            options={"feasibility_tolerance": 1e-7},
        )
        self.assertTrue(result.feasible, msg=result.raw_status)
        diagnostics = evaluate_incumbent(problem, result.x, support)
        self.assertTrue(diagnostics["numerically_feasible"])
        self.assertAlmostEqual(
            float(result.objective),
            float(diagnostics["upper_bound"]),
            places=10,
        )

    def test_auto_is_reproducible_and_branch_ready(self) -> None:
        problem = small_instance()
        options = {
            "random_samples": 12,
            "dfo_iterations": 3,
            "maximum_swap_evaluations": 8,
        }
        first = solve_incumbent(
            problem,
            problem.anchor,
            method="auto",
            restricted_solver="scipy",
            random_state=5,
            required_assets=(0,),
            forbidden_assets=(23,),
            options=options,
        )
        second = solve_incumbent(
            problem,
            problem.anchor,
            method="auto",
            restricted_solver="scipy",
            random_state=5,
            required_assets=(0,),
            forbidden_assets=(23,),
            options=options,
        )
        self.assertTrue(first.feasible)
        self.assertEqual(first.selectors[0], 1.0)
        self.assertEqual(first.selectors[23], 0.0)
        self.assertLessEqual(first.support.size, problem.k)
        np.testing.assert_allclose(first.weights, second.weights, atol=1e-10)
        self.assertAlmostEqual(first.upper_bound, second.upper_bound, places=10)
        restarted = solve_incumbent(
            problem,
            problem.anchor,
            method="fast",
            restricted_solver="scipy",
            warm_start=first,
            required_assets=(0,),
            forbidden_assets=(23,),
        )
        self.assertTrue(restarted.feasible)
        self.assertLessEqual(restarted.upper_bound, first.upper_bound + 1e-9)


if __name__ == "__main__":
    unittest.main()
