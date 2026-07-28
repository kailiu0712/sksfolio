"""Linearly constrained FISTA backend tests."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
import math
import unittest

import numpy as np

from sksfolio.relaxation import solve_relaxation

from tests._helpers import small_instance


def _budget_only_instance():
    problem = small_instance()
    budget_only = replace(
        problem,
        constraint_matrix=problem.C[:1, :].copy(),
        lower_bounds=np.array([1.0]),
        upper_bounds=np.array([1.0]),
        constraint_names=["budget"],
    )
    budget_only.validate()
    return budget_only


class FISTATests(unittest.TestCase):
    def test_public_backend_supports_both_pava_methods(self) -> None:
        problem = _budget_only_instance()
        results = {}
        for method in ("full_sort", "partial_sort"):
            results[method] = solve_relaxation(
                problem,
                backend="fista",
                pava=method,
                options={
                    "threads": 1,
                    "tolerance": 1e-7,
                    "max_iterations": 2_000,
                    "history_interval": 10,
                },
            )
            result = results[method]
            self.assertEqual(result.status, "converged")
            self.assertEqual(result.weights.shape, (problem.dimension,))
            self.assertLess(
                abs(float(np.sum(result.weights)) - 1.0),
                2e-8,
            )
            self.assertLessEqual(result.raw["violation"], 2e-8)
            self.assertEqual(result.raw["pava_backend"], method)
            self.assertEqual(
                result.raw["prox_oracle_used"],
                "exact_budget_scalar_brent",
            )
            self.assertTrue(result.raw["prox_exact_budget_fast_path"])
            self.assertGreater(result.raw["prox_calls"], 0)
            self.assertGreater(result.raw["prox_warm_starts"], 0)
            self.assertGreaterEqual(
                result.raw["pava_calls"],
                result.raw["prox_calls"],
            )
            self.assertTrue(result.raw["history"])
            self.assertIsNotNone(result.safe_dual_bound)
            self.assertTrue(result.dual_certificate.verify(problem))

        self.assertAlmostEqual(
            results["full_sort"].objective,
            results["partial_sort"].objective,
            places=7,
        )

    def test_additional_constraint_uses_general_dual_prox(self) -> None:
        problem = small_instance()
        result = solve_relaxation(
            problem,
            backend="fista",
            options={
                "threads": 1,
                "tolerance": 1e-4,
                "feasibility_tolerance": 1e-4,
                "max_iterations": 200,
                "history_interval": 10,
                "prox_tolerance": 1e-5,
                "prox_max_iterations": 100,
            },
        )
        self.assertEqual(result.status, "converged")
        self.assertEqual(
            result.raw["prox_oracle_used"],
            "row_scaled_dual_fista",
        )
        self.assertFalse(result.raw["prox_exact_budget_fast_path"])
        self.assertLessEqual(result.raw["violation"], 1e-4)
        self.assertTrue(result.dual_certificate.verify(problem))

    def test_adaptive_restart_can_be_disabled(self) -> None:
        result = solve_relaxation(
            _budget_only_instance(),
            backend="fista",
            options={
                "threads": 1,
                "adaptive_restart": False,
                "max_iterations": 5,
                "history_interval": 2,
            },
        )
        self.assertEqual(result.raw["variant"], "standard")
        self.assertEqual(result.raw["restarts"], 0)
        self.assertEqual(result.raw["restart_action"], "disabled")
        self.assertEqual(result.raw["restart_strategy"], "none")

    def test_all_outer_restart_strategies_return_safe_bounds(self) -> None:
        problem = _budget_only_instance()
        configurations = (
            ("none", {}),
            ("gradient", {}),
            ("function", {}),
            ("periodic", {"restart_period": 3}),
            ("hinder_lubin", {}),
            ("primal_dual_gap", {"restart_eta": math.e}),
        )
        for strategy, extra in configurations:
            with self.subTest(strategy=strategy):
                result = solve_relaxation(
                    problem,
                    backend="fista",
                    options={
                        "threads": 1,
                        "tolerance": 1e-7,
                        "max_iterations": 2_000,
                        "history_interval": 50,
                        "restart_strategy": strategy,
                        **extra,
                    },
                )
                self.assertEqual(result.status, "converged")
                self.assertEqual(
                    result.raw["restart_strategy"],
                    strategy,
                )
                self.assertTrue(
                    result.dual_certificate.verify(problem)
                )
                self.assertEqual(
                    len(result.raw["restart_iterations"]),
                    result.raw["restarts"],
                )
                restart_rows = [
                    row
                    for row in result.raw["history"]
                    if row["restart"]
                ]
                self.assertEqual(
                    len(restart_rows),
                    result.raw["restarts"],
                )

    def test_periodic_and_primal_dual_restart_bookkeeping(self) -> None:
        problem = _budget_only_instance()
        periodic = solve_relaxation(
            problem,
            backend="fista",
            options={
                "threads": 1,
                "tolerance": 1e-7,
                "max_iterations": 2_000,
                "history_interval": 100,
                "restart_strategy": "periodic",
                "restart_period": 3,
            },
        )
        self.assertTrue(periodic.raw["restart_iterations"])
        self.assertTrue(
            all(
                right - left == 3
                for left, right in zip(
                    [0] + periodic.raw["restart_iterations"][:-1],
                    periodic.raw["restart_iterations"],
                )
            )
        )

        gap = solve_relaxation(
            problem,
            backend="fista",
            options={
                "threads": 1,
                "tolerance": 1e-7,
                "max_iterations": 2_000,
                "history_interval": 100,
                "restart_strategy": "primal_dual_gap",
                "restart_eta": math.e,
            },
        )
        self.assertEqual(
            gap.raw["restart_checks"],
            gap.raw["iterations"],
        )
        self.assertEqual(
            gap.raw["restart_dual_evaluations"],
            gap.raw["iterations"],
        )
        self.assertTrue(
            gap.raw["restart_exact_prox_assumption_satisfied"]
        )
        for metric, threshold in zip(
            gap.raw["restart_metrics"],
            gap.raw["restart_thresholds"],
        ):
            self.assertLessEqual(metric, threshold + 1e-12)

    @unittest.skipUnless(
        importlib.util.find_spec("osqp") is not None,
        "OSQP is not installed",
    )
    def test_majorization_qp_is_a_selectable_prox(self) -> None:
        problem = small_instance()
        result = solve_relaxation(
            problem,
            backend="fista",
            options={
                "threads": 1,
                "prox_oracle": "majorization_qp",
                "tolerance": 1e-5,
                "feasibility_tolerance": 1e-5,
                "prox_tolerance": 1e-7,
                "majorization_max_iterations": 50_000,
                "max_iterations": 100,
                "history_interval": 10,
            },
        )
        self.assertEqual(result.status, "converged")
        self.assertEqual(
            result.raw["prox_oracle_used"],
            "majorization_qp_osqp",
        )
        self.assertEqual(result.raw["pava_calls"], 0)
        self.assertLessEqual(result.raw["violation"], 1e-5)
        self.assertIsNotNone(result.objective)
        self.assertTrue(result.dual_certificate.verify(problem))

    def test_plain_pava_rejects_linear_rows(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "requires no linear rows",
        ):
            solve_relaxation(
                small_instance(),
                backend="fista",
                options={"prox_oracle": "pava"},
            )


if __name__ == "__main__":
    unittest.main()
