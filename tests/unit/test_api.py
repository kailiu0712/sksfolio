"""Public API and backend-registry tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

import sksfolio
from sksfolio.relaxation import (
    BACKENDS,
    DEFAULT_BACKEND,
    DEFAULT_FISTA_PROX_ORACLE,
    DEFAULT_FISTA_RESTART,
    DEFAULT_PAVA,
    FISTA_PROX_ORACLES,
    FISTA_RESTART_STRATEGIES,
    PerspectiveRelaxation,
    available_backends,
    registered_backends,
    solve_relaxation,
)

from tests._helpers import small_instance


class APITests(unittest.TestCase):
    def test_backend_names_are_stable(self) -> None:
        self.assertEqual(
            BACKENDS,
            (
                "fista",
                "gurobi",
                "mosek",
                "jump",
                "gurobi.python",
                "gurobi.julia",
                "mosek.python",
                "mosek.julia",
                "jump.julia",
            ),
        )
        self.assertEqual(available_backends(), BACKENDS)
        self.assertEqual(registered_backends(), BACKENDS)

    def test_benchmark_selected_defaults_are_public(self) -> None:
        self.assertEqual(DEFAULT_BACKEND, "fista")
        self.assertEqual(DEFAULT_PAVA, "partial_sort")
        self.assertEqual(DEFAULT_FISTA_RESTART, "gradient")
        self.assertEqual(DEFAULT_FISTA_PROX_ORACLE, "auto")
        estimator = PerspectiveRelaxation()
        self.assertEqual(estimator.backend, DEFAULT_BACKEND)
        self.assertEqual(estimator.pava, DEFAULT_PAVA)

    def test_first_order_variant_choices_are_public(self) -> None:
        self.assertEqual(
            FISTA_PROX_ORACLES,
            (
                "auto",
                "pava",
                "budget",
                "dual_fista",
                "dual_lbfgs",
                "majorization_qp",
            ),
        )
        self.assertEqual(
            FISTA_RESTART_STRATEGIES,
            (
                "none",
                "gradient",
                "function",
                "periodic",
                "hinder_lubin",
                "primal_dual_gap",
            ),
        )
        self.assertIs(sksfolio.FISTA_PROX_ORACLES, FISTA_PROX_ORACLES)
        self.assertIs(
            sksfolio.FISTA_RESTART_STRATEGIES,
            FISTA_RESTART_STRATEGIES,
        )

    def test_fista_defaults_are_forwarded_explicitly(self) -> None:
        supplied = {}

        def fake_solver(instance, options):
            supplied.update(options)
            return {
                "status": "iteration_limit",
                "x": None,
                "has_solution": False,
            }

        with patch(
            "sksfolio.relaxation.api._backend_solver",
            return_value=fake_solver,
        ):
            result = solve_relaxation(small_instance())
        self.assertEqual(result.raw["backend"], "fista")
        self.assertEqual(
            supplied["restart_strategy"],
            DEFAULT_FISTA_RESTART,
        )
        self.assertEqual(
            supplied["prox_oracle"],
            DEFAULT_FISTA_PROX_ORACLE,
        )
        self.assertEqual(supplied["pava_backend"], DEFAULT_PAVA)

    def test_short_commercial_names_select_native_python(self) -> None:
        unavailable = {
            "status": "unavailable",
            "x": None,
            "has_solution": False,
        }
        with patch(
            "sksfolio.relaxation.api._backend_solver",
            return_value=lambda instance, options: unavailable,
        ):
            for backend in ("gurobi", "mosek"):
                with self.subTest(backend=backend):
                    result = solve_relaxation(
                        small_instance(),
                        backend,
                    )
                    self.assertEqual(
                        result.raw["canonical_backend"],
                        f"{backend}.python",
                    )
                    self.assertEqual(
                        result.raw["language"],
                        "python",
                    )
                    self.assertEqual(result.raw["solver"], backend)

    def test_short_jump_name_selects_generic_julia_backend(self) -> None:
        unavailable = {
            "status": "unavailable",
            "x": None,
            "has_solution": False,
            "solver": "clarabel",
        }
        with patch(
            "sksfolio.relaxation.api._backend_solver",
            return_value=lambda instance, options: unavailable,
        ):
            result = solve_relaxation(small_instance(), "jump")
        self.assertEqual(result.raw["canonical_backend"], "jump.julia")
        self.assertEqual(result.raw["language"], "julia")
        self.assertEqual(result.raw["solver"], "clarabel")

    def test_named_false_disables_commercial_warm_start(self) -> None:
        for backend in ("gurobi", "mosek", "jump"):
            with self.subTest(backend=backend):
                captured = {}

                def fake_solver(instance, options):
                    captured.update(options)
                    return {
                        "status": "unavailable",
                        "x": None,
                        "has_solution": False,
                    }

                with patch(
                    "sksfolio.relaxation.api._backend_solver",
                    return_value=fake_solver,
                ):
                    solve_relaxation(
                        small_instance(),
                        backend,
                        warm_start=False,
                    )
                self.assertIs(captured["warm_start"], False)

    def test_public_diagnostics_use_the_solver_tolerance(self) -> None:
        problem = small_instance()
        weights = np.zeros(problem.dimension)
        weights[0] = 1.0 + 5e-7

        def fake_solver(instance, options):
            return {
                "status": "optimal",
                "x": weights,
                "has_solution": True,
                "effective_tolerance": 1e-6,
            }

        with patch(
            "sksfolio.relaxation.api._backend_solver",
            return_value=fake_solver,
        ):
            result = solve_relaxation(problem, "gurobi")
        self.assertIsNotNone(result.objective)
        self.assertEqual(
            result.raw["diagnostic_domain_tolerance"],
            1e-6,
        )
        self.assertGreater(
            result.raw["diagnostics"]["violations"]["upper_box"],
            0.0,
        )
        self.assertTrue(result.dual_certificate.verify(problem))

    def test_nonfinite_solver_bound_is_not_exposed(self) -> None:
        with patch(
            "sksfolio.relaxation.api._backend_solver",
            return_value=lambda instance, options: {
                "status": "time_limit",
                "x": None,
                "has_solution": False,
                "solver_details": {"objective_bound": np.nan},
            },
        ):
            result = solve_relaxation(small_instance(), "mosek")
        self.assertIsNone(result.solver_objective_bound)
        self.assertNotIn("solver_objective_bound", result.raw)

    def test_estimator_style_wrapper(self) -> None:
        problem = small_instance()
        estimator = PerspectiveRelaxation(
            backend="pdhg",
            variant="fixed",
            pava="partial_sort",
            solver_params={
                "threads": 1,
                "max_iterations": 20,
                "check_interval": 10,
                "min_epoch": 10,
                "max_epoch": 20,
            },
        ).fit(problem)
        self.assertEqual(estimator.weights_.shape, (problem.dimension,))
        self.assertTrue(np.all(np.isfinite(estimator.weights_)))
        self.assertIsNotNone(estimator.result_.safe_dual_bound)

    def test_unknown_backend_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "backend"):
            solve_relaxation(small_instance(), "newton")

    def test_pava_choice_has_one_public_location(self) -> None:
        with self.assertRaisesRegex(ValueError, "pava="):
            solve_relaxation(
                small_instance(),
                "pdhg",
                pava="full_sort",
                options={"pava_backend": "partial_sort"},
            )

    def test_missing_julia_runtime_is_normalized(self) -> None:
        with patch(
            "sksfolio.relaxation._julia_runner._load_module",
            side_effect=RuntimeError("juliacall is unavailable"),
        ):
            result = solve_relaxation(
                small_instance(),
                "gurobi.julia",
            )
        self.assertEqual(result.status, "unavailable")
        self.assertFalse(result.raw["has_solution"])


if __name__ == "__main__":
    unittest.main()
