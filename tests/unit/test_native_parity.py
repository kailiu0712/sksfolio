"""Numerical parity checks for the generated-C first-order solvers."""

from __future__ import annotations

from dataclasses import replace
import math
import unittest

import numpy as np

from sksfolio.relaxation import evaluate_solution, solve_relaxation
from sksfolio.relaxation.native import native_available
from sksfolio.relaxation.pdhg import PDHG_VARIANTS

from tests._helpers import small_instance


_RTOL = 2e-13
_ATOL = 2e-13


def _budget_only_instance():
    problem = small_instance(dimension=12)
    budget_only = replace(
        problem,
        constraint_matrix=problem.C[:1, :].copy(),
        lower_bounds=np.array([1.0]),
        upper_bounds=np.array([1.0]),
        constraint_names=["budget"],
    )
    budget_only.validate()
    return budget_only


class NativeParityTests(unittest.TestCase):
    """Keep the Python reference and compiled implementations in lockstep."""

    def assert_trajectory_parity(self, python, native) -> None:
        python_history = python.raw["history"]
        native_history = native.raw["history"]
        self.assertEqual(len(python_history), len(native_history))
        for python_row, native_row in zip(
            python_history,
            native_history,
            strict=True,
        ):
            self.assertEqual(python_row.keys(), native_row.keys())
            for key, expected in python_row.items():
                if "seconds" in key or key.startswith("elapsed"):
                    continue
                actual = native_row[key]
                if isinstance(expected, (float, np.floating)):
                    if math.isnan(float(expected)):
                        self.assertTrue(math.isnan(float(actual)))
                    else:
                        np.testing.assert_allclose(
                            actual,
                            expected,
                            rtol=_RTOL,
                            atol=_ATOL,
                            err_msg=f"history field {key!r}",
                        )
                elif isinstance(expected, np.ndarray):
                    np.testing.assert_allclose(
                        actual,
                        expected,
                        rtol=_RTOL,
                        atol=_ATOL,
                        err_msg=f"history field {key!r}",
                    )
                else:
                    self.assertEqual(actual, expected, key)

    def assert_certificate_is_safe(self, problem, result) -> None:
        certificate = result.dual_certificate
        self.assertIsNotNone(certificate)
        self.assertTrue(certificate.verify(problem, tolerance=2e-12))
        anchor_objective = evaluate_solution(
            problem,
            problem.anchor,
        )["objective"]
        self.assertIsNotNone(anchor_objective)
        self.assertLessEqual(
            certificate.lower_bound,
            float(anchor_objective) + 2e-11,
        )

    def assert_result_parity(self, problem, python, native) -> None:
        self.assertEqual(python.status, native.status)
        self.assertEqual(python.raw["iterations"], native.raw["iterations"])
        self.assertEqual(python.raw["implementation"], "python")
        self.assertEqual(native.raw["implementation"], "native")
        self.assertFalse(python.raw["native_solver_core"])
        self.assertTrue(native.raw["native_solver_core"])
        np.testing.assert_allclose(
            native.weights,
            python.weights,
            rtol=_RTOL,
            atol=_ATOL,
        )
        for key in (
            "external_objective",
            "residual",
            "relative_residual",
            "violation",
            "dual_bound",
            "best_dual_lower_bound",
        ):
            expected = python.raw.get(key)
            actual = native.raw.get(key)
            if expected is None or actual is None:
                self.assertIs(expected, actual, key)
            else:
                np.testing.assert_allclose(
                    actual,
                    expected,
                    rtol=_RTOL,
                    atol=_ATOL,
                    err_msg=key,
                )
        for key in (
            "dual_bound_factor",
            "dual_bound_constraint_original",
        ):
            np.testing.assert_allclose(
                native.raw[key],
                python.raw[key],
                rtol=_RTOL,
                atol=_ATOL,
                err_msg=key,
            )

        self.assert_trajectory_parity(python, native)
        self.assert_certificate_is_safe(problem, python)
        self.assert_certificate_is_safe(problem, native)
        np.testing.assert_allclose(
            native.dual_certificate.factor_multiplier,
            python.dual_certificate.factor_multiplier,
            rtol=_RTOL,
            atol=_ATOL,
        )
        np.testing.assert_allclose(
            native.dual_certificate.constraint_multiplier,
            python.dual_certificate.constraint_multiplier,
            rtol=_RTOL,
            atol=_ATOL,
        )

    @unittest.skipUnless(
        native_available("fista"),
        "compiled FISTA extension is not installed",
    )
    def test_fista_auto_lbfgs_and_budget_match_python(self) -> None:
        general = small_instance(dimension=12)
        configurations = (
            ("auto", general, {}),
            (
                "explicit_lbfgs",
                general,
                {"prox_oracle": "dual_lbfgs"},
            ),
            (
                "exact_budget",
                _budget_only_instance(),
                {"prox_oracle": "budget"},
            ),
        )
        common = {
            "threads": 1,
            "max_iterations": 6,
            "history_interval": 1,
            "tolerance": 1e-14,
            "feasibility_tolerance": 1e-14,
            "prox_tolerance": 1e-10,
            "prox_max_iterations": 100,
            "restart_strategy": "gradient",
        }
        for name, problem, extra in configurations:
            with self.subTest(mode=name):
                python = solve_relaxation(
                    problem,
                    "fista",
                    implementation="python",
                    options={**common, **extra},
                )
                native = solve_relaxation(
                    problem,
                    "fista",
                    implementation="native",
                    options={**common, **extra},
                )
                self.assert_result_parity(problem, python, native)
                if name in {"auto", "explicit_lbfgs"}:
                    self.assertEqual(
                        native.raw["prox_oracle_used"],
                        "row_scaled_dual_lbfgsb",
                    )
                else:
                    self.assertEqual(
                        native.raw["prox_oracle_used"],
                        "exact_budget_scalar_brent",
                    )

    @unittest.skipUnless(
        native_available("pdhg"),
        "compiled PDHG extension is not installed",
    )
    def test_every_pdhg_variant_matches_python(self) -> None:
        problem = small_instance(dimension=12)
        options = {
            "threads": 1,
            "max_iterations": 6,
            "check_interval": 1,
            "min_epoch": 2,
            "max_epoch": 4,
            "tolerance": 1e-14,
            "feasibility_tolerance": 1e-14,
            "norm_iterations": 10,
        }
        for variant in PDHG_VARIANTS:
            with self.subTest(variant=variant):
                python = solve_relaxation(
                    problem,
                    "pdhg",
                    variant=variant,
                    implementation="python",
                    options=options,
                )
                native = solve_relaxation(
                    problem,
                    "pdhg",
                    variant=variant,
                    implementation="native",
                    options=options,
                )
                self.assert_result_parity(problem, python, native)
                self.assertEqual(python.raw["variant"], variant)
                self.assertEqual(native.raw["variant"], variant)

    @unittest.skipUnless(
        native_available("scsdg"),
        "compiled SC-SDG extension is not installed",
    )
    def test_scsdg_fixed_and_line_search_modes_match_python(self) -> None:
        problem = small_instance(dimension=12)
        common = {
            "threads": 1,
            "max_iterations": 6,
            "check_interval": 1,
            "restart": True,
            "restart_check_interval": 1,
            "min_restart_iterations": 1,
            "step_ratio": 0.3,
            "tolerance": 1e-14,
            "feasibility_tolerance": 1e-14,
            "operator_norm_mode": "gram",
            "lbfgs_variant": "off",
        }
        configurations = (
            ("fixed", {"line_search": False}),
            (
                "operator",
                {
                    "line_search": True,
                    "line_search_mode": "operator",
                    "line_search_initial_scale": 2.0,
                    "line_search_growth": 1.0,
                    "line_search_max_scale": 2.0,
                },
            ),
            (
                "majorization",
                {
                    "line_search": True,
                    "line_search_mode": "majorization",
                },
            ),
        )
        for mode, extra in configurations:
            with self.subTest(mode=mode):
                options = {**common, **extra}
                python = solve_relaxation(
                    problem,
                    "scsdg",
                    implementation="python",
                    options=options,
                )
                native = solve_relaxation(
                    problem,
                    "scsdg",
                    implementation="native",
                    options=options,
                )
                self.assert_result_parity(problem, python, native)
                self.assertEqual(
                    native.raw["line_search_enabled"],
                    mode != "fixed",
                )


if __name__ == "__main__":
    unittest.main()
