"""Public API tests for the two corrected algorithms."""

from __future__ import annotations

import unittest

import numpy as np

import sksfolio
from sksfolio.relaxation import (
    BACKENDS,
    CORRECTED_ALGORITHMS,
    COMMERCIAL_BACKENDS,
    DEFAULT_BACKEND,
    PerspectiveRelaxation,
    available_backends,
    registered_backends,
    solve_relaxation,
)

from tests._helpers import small_instance


class APITests(unittest.TestCase):
    def test_algorithm_registry_includes_hybrid_and_commercial_backends(self) -> None:
        expected = ("corrected_fista", "corrected_lbfgs")
        self.assertEqual(CORRECTED_ALGORITHMS, expected)
        backends = expected + ("hybrid_newton",) + COMMERCIAL_BACKENDS
        self.assertEqual(BACKENDS, backends)
        self.assertEqual(available_backends(), backends)
        self.assertEqual(registered_backends(), backends)
        self.assertEqual(DEFAULT_BACKEND, "corrected_lbfgs")
        self.assertIs(sksfolio.CORRECTED_ALGORITHMS, CORRECTED_ALGORITHMS)

    def test_both_algorithms_return_safe_certified_results(self) -> None:
        problem = small_instance()
        results = {}
        for algorithm in CORRECTED_ALGORITHMS + ("hybrid_newton",):
            with self.subTest(algorithm=algorithm):
                result = solve_relaxation(
                    problem,
                    algorithm,
                    options={
                        "threads": 1,
                        "tolerance": 1e-5,
                        "feasibility_tolerance": 1e-5,
                        "max_iterations": 2_000,
                        "prox_tolerance": 1e-7,
                        "prox_max_iterations": 2_000,
                    },
                )
                results[algorithm] = result
                self.assertEqual(result.status, "converged")
                self.assertEqual(result.raw["algorithm"], algorithm)
                self.assertEqual(result.weights.shape, (problem.dimension,))
                self.assertTrue(np.all(np.isfinite(result.weights)))
                self.assertTrue(result.dual_certificate.verify(problem))
                self.assertLessEqual(result.raw["violation"], 1e-5)
        self.assertAlmostEqual(
            results["corrected_fista"].objective,
            results["corrected_lbfgs"].objective,
            places=5,
        )

    def test_algorithm_controls_cannot_be_overridden(self) -> None:
        for key in (
            "algorithm_variant",
            "prox_oracle",
            "pava_backend",
            "implementation",
        ):
            with self.subTest(key=key), self.assertRaisesRegex(
                ValueError, "controlled by backend"
            ):
                solve_relaxation(
                    small_instance(),
                    "corrected_fista",
                    options={key: "anything"},
                )

    def test_unknown_or_removed_backend_is_rejected(self) -> None:
        for backend in ("fista", "pdhg", "gurobi", "mosek", "jump"):
            with self.subTest(backend=backend), self.assertRaisesRegex(
                ValueError, "backend"
            ):
                solve_relaxation(small_instance(), backend)

    def test_estimator_can_reuse_corrected_restart_state(self) -> None:
        problem = small_instance()
        estimator = PerspectiveRelaxation(
            backend="corrected_lbfgs",
            solver_params={
                "threads": 1,
                "tolerance": 1e-5,
                "max_iterations": 2_000,
            },
        ).fit(problem)
        first_objective = estimator.result_.objective
        estimator.fit(problem, warm_start=True)
        self.assertTrue(estimator.result_.raw["warm_start_used"])
        self.assertAlmostEqual(
            estimator.result_.objective,
            first_objective,
            places=4,
        )
        self.assertEqual(estimator.state_.backend, "corrected_lbfgs")


if __name__ == "__main__":
    unittest.main()
