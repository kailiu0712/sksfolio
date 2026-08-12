"""Correctness checks for the restarted SC-SDG backend."""

from __future__ import annotations

import unittest

import numpy as np

from sksfolio.relaxation import solve_relaxation
from sksfolio.relaxation.pdhg.solver import _prepare_problem
from sksfolio.relaxation.scsdg.solver import (
    _Point,
    _operator_setup,
    _product_value,
    _settings,
    _smoothed_oracle,
    solve_scsdg,
)

from tests._helpers import small_instance


class SmoothedGapTests(unittest.TestCase):
    def test_fixed_gap_is_nonnegative(self) -> None:
        instance = small_instance()
        settings, _ = _settings(
            {
                "step_ratio": 0.3,
                "constraint_dual_weight": 2.0,
            }
        )
        problem = _prepare_problem(instance)
        operator = _operator_setup(problem, settings)
        beta_norm = float(operator["beta_norm"])
        root = (
            beta_norm
            * np.sqrt(
                settings.target_cbar
                * settings.theta_parameter
            )
            / settings.continuation_offset
        )
        beta_x = root * settings.step_ratio
        beta_y = root / settings.step_ratio
        point = _Point(
            problem.anchor.copy(),
            np.linspace(-0.02, 0.03, problem.factors),
            np.array([0.01, -0.015]),
        )
        oracle = _smoothed_oracle(
            problem,
            point,
            beta_x,
            beta_y,
            np.sqrt(settings.constraint_weight),
            settings.pava_backend,
            evaluate_gap=True,
        )
        self.assertIsNotNone(oracle.gap)
        self.assertGreaterEqual(float(oracle.gap), 0.0)
        self.assertGreaterEqual(oracle.residual, 0.0)
        self.assertEqual(oracle.gradient.x.shape, point.x.shape)
        self.assertEqual(oracle.gradient.p.shape, point.p.shape)
        self.assertEqual(oracle.gradient.q.shape, point.q.shape)

    def test_smooth_component_gradient_matches_finite_difference(
        self,
    ) -> None:
        instance = small_instance()
        settings, _ = _settings(
            {
                "step_ratio": 0.7,
                "constraint_dual_weight": 1.5,
            }
        )
        problem = _prepare_problem(instance)
        operator = _operator_setup(problem, settings)
        beta_norm = float(operator["beta_norm"])
        root = (
            beta_norm
            * np.sqrt(
                settings.target_cbar
                * settings.theta_parameter
            )
            / settings.continuation_offset
        )
        beta_x = root * settings.step_ratio
        beta_y = root / settings.step_ratio
        constraint_scale = np.sqrt(
            float(operator["constraint_weight"])
        )
        point = _Point(
            0.97 * problem.anchor.copy(),
            np.linspace(-0.01, 0.02, problem.factors),
            np.array([0.015, -0.012]),
        )
        oracle = _smoothed_oracle(
            problem,
            point,
            beta_x,
            beta_y,
            constraint_scale,
            settings.pava_backend,
            evaluate_gap=True,
        )

        def smooth_value(candidate: _Point) -> float:
            evaluated = _smoothed_oracle(
                problem,
                candidate,
                beta_x,
                beta_y,
                constraint_scale,
                settings.pava_backend,
                evaluate_gap=True,
            )
            return float(evaluated.gap) - _product_value(
                problem,
                candidate,
                constraint_scale,
            )

        epsilon = 2e-7
        coordinates = (
            ("x", 3, oracle.gradient.x[3]),
            ("p", 1, oracle.gradient.p[1]),
            ("q", 0, oracle.gradient.q[0]),
        )
        for block, index, expected in coordinates:
            plus = point.copy()
            minus = point.copy()
            getattr(plus, block)[index] += epsilon
            getattr(minus, block)[index] -= epsilon
            finite_difference = (
                smooth_value(plus) - smooth_value(minus)
            ) / (2.0 * epsilon)
            self.assertAlmostEqual(
                finite_difference,
                float(expected),
                delta=2e-6,
            )

    def test_full_and_partial_pava_generate_same_trajectory(self) -> None:
        instance = small_instance()
        common = {
            "threads": 1,
            "max_iterations": 12,
            "check_interval": 4,
            "restart": False,
            "step_ratio": 0.3,
        }
        full = solve_scsdg(
            instance,
            {**common, "pava_backend": "full_sort"},
        )
        partial = solve_scsdg(
            instance,
            {**common, "pava_backend": "partial_sort"},
        )
        np.testing.assert_allclose(
            full["x"],
            partial["x"],
            rtol=2e-12,
            atol=2e-12,
        )
        self.assertAlmostEqual(
            full["smoothed_gap"],
            partial["smoothed_gap"],
            places=12,
        )

    def test_public_backend_returns_verifiable_safe_bound(self) -> None:
        instance = small_instance()
        result = solve_relaxation(
            instance,
            backend="scsdg",
            pava="partial_sort",
            options={
                "threads": 1,
                "max_iterations": 100,
                "check_interval": 10,
                "restart_check_interval": 5,
                "step_ratio": 0.3,
            },
        )
        self.assertEqual(result.weights.shape, (instance.dimension,))
        self.assertTrue(np.all(np.isfinite(result.weights)))
        self.assertIsNotNone(result.safe_dual_bound)
        self.assertIsNotNone(result.dual_certificate)
        self.assertTrue(result.dual_certificate.verify(instance))
        self.assertGreater(result.raw["pava_calls"], 0)
        self.assertGreater(result.raw["smoothed_gap_evaluations"], 0)
        self.assertEqual(
            result.raw["operator_norm_kind"],
            "exact_dual_gram",
        )
        self.assertEqual(
            result.raw["line_search_mode_requested"],
            "auto",
        )
        self.assertEqual(
            result.raw["line_search_mode_resolved"],
            "majorization",
        )

    def test_halving_restart_resets_continuation(self) -> None:
        result = solve_scsdg(
            small_instance(),
            {
                "threads": 1,
                "max_iterations": 100,
                "check_interval": 5,
                "restart_check_interval": 1,
                "step_ratio": 0.3,
            },
        )
        self.assertGreater(result["restarts"], 0)
        self.assertEqual(
            result["restart_certificate"],
            "fixed_beta_self_centered_smoothed_gap_halving",
        )
        self.assertTrue(
            any(point["restart"] for point in result["history"][1:])
        )

    def test_operator_line_search_backtracks_and_is_accounted(self) -> None:
        instance = small_instance()
        result = solve_scsdg(
            instance,
            {
                "threads": 1,
                "max_iterations": 20,
                "check_interval": 5,
                "restart": False,
                "step_ratio": 0.3,
                "line_search": True,
                "line_search_mode": "operator",
                "line_search_initial_scale": 2.0,
                "line_search_growth": 1.0,
                "line_search_max_scale": 2.0,
            },
        )
        self.assertGreater(result["line_search_backtracks"], 0)
        self.assertEqual(
            result["line_search_trials"],
            result["iterations"] + result["line_search_backtracks"],
        )
        self.assertEqual(
            result["pava_calls"],
            result["iterations"]
            + result["line_search_trials"]
            + result["smoothed_gap_evaluations"],
        )
        self.assertLessEqual(
            result[
                "line_search_maximum_accepted_curvature_ratio"
            ],
            result["line_search_safety"] + 1e-10,
        )
        self.assertFalse(result["theorem_step_certified"])
        self.assertTrue(result["operator_norm_certified"])

    def test_majorization_line_search_preserves_safe_certificate(
        self,
    ) -> None:
        instance = small_instance()
        result = solve_relaxation(
            instance,
            backend="scsdg",
            options={
                "threads": 1,
                "max_iterations": 20,
                "check_interval": 5,
                "restart": False,
                "step_ratio": 0.3,
                "line_search": True,
                "line_search_mode": "majorization",
            },
        )
        self.assertEqual(
            result.raw["line_search_mode"],
            "smooth_scsdg_weighted_quadratic_majorization",
        )
        self.assertGreater(
            result.raw["line_search_minimum_majorization_slack"],
            -1e-10,
        )
        self.assertTrue(result.dual_certificate.verify(instance))

    def test_fixed_step_mode_keeps_the_paper_step_certificate(
        self,
    ) -> None:
        result = solve_scsdg(
            small_instance(),
            {
                "threads": 1,
                "max_iterations": 12,
                "check_interval": 4,
                "restart": False,
                "line_search": False,
                "step_ratio": 0.3,
            },
        )
        self.assertEqual(result["line_search_mode"], "disabled")
        self.assertEqual(result["line_search_trials"], 0)
        self.assertEqual(result["line_search_backtracks"], 0)
        self.assertTrue(result["theorem_step_certified"])
        self.assertEqual(
            result["pava_calls"],
            2 * result["iterations"]
            + result["smoothed_gap_evaluations"],
        )

    def test_all_lbfgs_variants_call_the_polisher_and_flag_rate(
        self,
    ) -> None:
        instance = small_instance()
        common = {
            "threads": 1,
            "max_iterations": 25,
            "check_interval": 5,
            "restart_check_interval": 1,
            "step_ratio": 0.3,
            "lbfgs_min_function_evaluations": 2,
            "lbfgs_max_function_evaluations": 4,
        }
        for variant in (
            "paper",
            "restart_reference",
            "restart_safe",
        ):
            with self.subTest(variant=variant):
                result = solve_scsdg(
                    instance,
                    {**common, "lbfgs_variant": variant},
                )
                self.assertEqual(result["lbfgs_variant"], variant)
                self.assertTrue(result["lbfgs_enabled"])
                self.assertGreater(result["lbfgs_calls"], 0)
                self.assertEqual(
                    result["lbfgs_calls"],
                    result["lbfgs_accepted_calls"]
                    + result["lbfgs_rejected_calls"],
                )
                self.assertGreater(
                    result["lbfgs_function_evaluations"],
                    0,
                )
                self.assertGreater(result["lbfgs_pava_calls"], 0)
                self.assertFalse(result["theorem_step_certified"])
                self.assertFalse(
                    result["paper_rate_applies_without_extension"]
                )
                self.assertTrue(
                    result[
                        "lbfgs_is_paper_extension_without_rate_proof"
                    ]
                )
                self.assertTrue(
                    any(
                        row["lbfgs_triggered"]
                        for row in result["history"]
                    )
                )
                if variant == "paper":
                    self.assertGreater(
                        result["lbfgs_anticipated_calls"],
                        0,
                    )
                else:
                    self.assertEqual(
                        result["lbfgs_anticipated_calls"],
                        0,
                    )
                    self.assertEqual(
                        result["lbfgs_restart_calls"],
                        result["lbfgs_calls"],
                    )

    def test_safeguarded_lbfgs_keeps_the_public_dual_bound_safe(
        self,
    ) -> None:
        instance = small_instance()
        result = solve_relaxation(
            instance,
            backend="scsdg",
            options={
                "threads": 1,
                "max_iterations": 25,
                "check_interval": 5,
                "restart_check_interval": 1,
                "step_ratio": 0.3,
                "lbfgs_variant": "restart_safe",
                "lbfgs_min_function_evaluations": 2,
                "lbfgs_max_function_evaluations": 4,
            },
        )
        self.assertGreater(result.raw["lbfgs_calls"], 0)
        self.assertGreater(result.raw["lbfgs_accepted_calls"], 0)
        self.assertEqual(
            result.raw["lbfgs_acceptance_rule"],
            "restart_fixed_beta_gap_nonincrease",
        )
        self.assertIsNotNone(result.safe_dual_bound)
        self.assertTrue(result.dual_certificate.verify(instance))

    def test_line_search_options_are_validated(self) -> None:
        invalid_options = (
            {"line_search_mode": "unknown"},
            {"line_search_growth": 0.9},
            {"line_search_shrink": 1.0},
            {"line_search_safety": 0.0},
            {"line_search_tolerance": -1.0},
            {"line_search_max_backtracks": 0},
            {
                "line_search_initial_scale": 2.0,
                "line_search_max_scale": 1.0,
            },
        )
        for options in invalid_options:
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    _settings(options)


if __name__ == "__main__":
    unittest.main()
