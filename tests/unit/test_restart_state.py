"""Portable restart-state and compiled-dispatch tests."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse

from sksfolio.relaxation import (
    MarkowitzInstance,
    PerspectiveRelaxation,
    RelaxationState,
    native_available,
    solve_relaxation,
)
from sksfolio.relaxation.pdhg.safe_dual import evaluate_dual_bound

from tests._helpers import small_instance


def _quick_options(backend: str) -> dict[str, object]:
    common: dict[str, object] = {
        "threads": 1,
        "max_iterations": 3,
    }
    if backend == "fista":
        common.update(
            {
                "history_interval": 1,
                "prox_max_iterations": 20,
            }
        )
    elif backend == "pdhg":
        common.update(
            {
                "check_interval": 1,
                "min_epoch": 1,
                "max_epoch": 3,
            }
        )
    else:
        common.update(
            {
                "check_interval": 1,
                "restart_check_interval": 1,
            }
        )
    return common


class RelaxationStateTests(unittest.TestCase):
    def test_npz_round_trip_preserves_portable_and_private_fields(
        self,
    ) -> None:
        state = RelaxationState(
            backend="fista",
            implementation="native",
            dimension=4,
            k=2,
            constraint_ids=("budget", "sector"),
            x=np.array([0.1, 0.2, 0.3, 0.4]),
            factor_dual=np.array([0.25, -0.75]),
            constraint_dual=np.array([3.0, -2.0]),
            arrays={"momentum_x": np.arange(4, dtype=float)},
            scalars={
                "lipschitz": 7.5,
                "fresh_restart_epoch_recommended": True,
                "optional": None,
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            path = state.save(Path(directory) / "node-state.npz")
            restored = RelaxationState.load(path)

        self.assertEqual(restored.backend, state.backend)
        self.assertEqual(restored.implementation, state.implementation)
        self.assertEqual(restored.dimension, state.dimension)
        self.assertEqual(restored.k, state.k)
        self.assertEqual(restored.constraint_ids, state.constraint_ids)
        self.assertEqual(restored.scalars, state.scalars)
        np.testing.assert_array_equal(restored.x, state.x)
        np.testing.assert_array_equal(
            restored.factor_dual,
            state.factor_dual,
        )
        np.testing.assert_array_equal(
            restored.constraint_dual,
            state.constraint_dual,
        )
        np.testing.assert_array_equal(
            restored.arrays["momentum_x"],
            state.arrays["momentum_x"],
        )

    def test_constraint_duals_are_remapped_by_name_for_child_nodes(
        self,
    ) -> None:
        state = RelaxationState(
            backend="pdhg",
            implementation="python",
            dimension=4,
            k=2,
            constraint_ids=("budget", "sector", "old_branch"),
            x=np.full(4, 0.25),
            constraint_dual=np.array([2.0, -3.0, 9.0]),
        )
        child = SimpleNamespace(
            dimension=4,
            k=2,
            constraint_names=("new_branch", "sector", "budget"),
        )
        np.testing.assert_array_equal(
            state.mapped_constraint_dual(child),
            np.array([0.0, -3.0, 2.0]),
        )
        np.testing.assert_array_equal(
            state.compatible_primal(child),
            state.x,
        )

    def test_python_solvers_accept_a_prior_result_and_restart_momentum(
        self,
    ) -> None:
        problem = small_instance(dimension=12)
        for backend in ("fista", "pdhg", "scsdg"):
            with self.subTest(backend=backend):
                options = _quick_options(backend)
                first = solve_relaxation(
                    problem,
                    backend,
                    implementation="python",
                    options=options,
                )
                restarted = solve_relaxation(
                    problem,
                    backend,
                    implementation="python",
                    warm_start=first,
                    options=options,
                )

                self.assertIsNotNone(first.state)
                self.assertTrue(restarted.raw["warm_start_used"])
                self.assertTrue(
                    restarted.raw["warm_start_exact_structure"]
                )
                self.assertFalse(
                    restarted.raw["warm_start_acceleration_resumed"]
                )
                self.assertEqual(restarted.raw["implementation"], "python")
                self.assertFalse(restarted.raw["native_solver_core"])
                self.assertEqual(restarted.state.backend, backend)
                self.assertEqual(
                    restarted.state.constraint_ids,
                    tuple(problem.constraint_names),
                )
                self.assertTrue(np.all(np.isfinite(restarted.weights)))

    def test_public_state_retains_the_best_safe_certificate(self) -> None:
        problem = small_instance(dimension=12)
        for backend in ("fista", "scsdg"):
            with self.subTest(backend=backend):
                result = solve_relaxation(
                    problem,
                    backend,
                    implementation="python",
                    options=_quick_options(backend),
                )
                state = result.state
                recomputed = evaluate_dual_bound(
                    problem,
                    state.factor_dual,
                    state.constraint_dual,
                )["dual_bound"]
                self.assertIsNotNone(recomputed)
                self.assertAlmostEqual(
                    recomputed,
                    result.safe_dual_bound,
                    places=10,
                )
                self.assertIn("current_factor_dual", state.arrays)
                self.assertIn("current_constraint_dual", state.arrays)

    def test_fista_child_start_is_not_an_infeasible_upper_bound(self) -> None:
        problem = small_instance(dimension=12)
        options = _quick_options("fista")
        parent = solve_relaxation(
            problem,
            "fista",
            implementation="python",
            options=options,
        )
        branch_index = int(np.argmax(parent.weights))
        branch_row = np.zeros(problem.dimension)
        branch_row[branch_index] = 1.0
        branch_upper = 0.5 * float(parent.weights[branch_index])
        child = MarkowitzInstance(
            factor_loadings=problem.B,
            expected_returns=problem.mu,
            constraint_matrix=sparse.vstack(
                [problem.C, sparse.csr_matrix(branch_row)]
            ).tocsr(),
            lower_bounds=np.r_[problem.lower, -np.inf],
            upper_bounds=np.r_[problem.upper, branch_upper],
            feasible_anchor=parent.weights,
            constraint_names=[*problem.constraint_names, "branch_upper"],
            k=problem.k,
            perspective_weight=problem.perspective_weight,
            return_reward=problem.return_reward,
            anchor_must_be_feasible=False,
        )
        child.validate()

        result = solve_relaxation(
            child,
            "fista",
            implementation="python",
            warm_start=parent,
            options={**options, "resume_acceleration": True},
        )

        self.assertIsNone(result.raw["anchor_primal_upper_bound"])
        self.assertIsNone(result.raw["anchor_primal_dual_gap"])
        self.assertFalse(result.raw["anchor_is_numerically_feasible"])
        self.assertFalse(result.raw["warm_start_acceleration_resumed"])
        self.assertGreaterEqual(
            result.safe_dual_bound + 1e-12,
            parent.safe_dual_bound,
        )

    def test_api_enriches_x_only_commercial_state_and_warm_metadata(
        self,
    ) -> None:
        problem = small_instance(dimension=12)
        x_only = RelaxationState(
            backend="gurobi",
            implementation="python",
            dimension=problem.dimension,
            k=problem.k,
            constraint_ids=tuple(problem.constraint_names),
            x=problem.anchor,
        )

        def fake_solver(instance, options):
            return {
                "status": "optimal",
                "x": problem.anchor.copy(),
                "has_solution": True,
                "effective_tolerance": 1e-8,
                "solver_details": {"warm_start_used": True},
                "restart_state": x_only.to_dict(copy=False),
            }

        with patch(
            "sksfolio.relaxation.api._backend_solver",
            return_value=fake_solver,
        ):
            result = solve_relaxation(problem, "gurobi")

        self.assertTrue(result.raw["warm_start_used"])
        self.assertIsNotNone(result.state.factor_dual)
        self.assertIsNotNone(result.state.constraint_dual)
        self.assertAlmostEqual(
            evaluate_dual_bound(
                problem,
                result.state.factor_dual,
                result.state.constraint_dual,
            )["dual_bound"],
            result.safe_dual_bound,
            places=10,
        )
        self.assertTrue(result.raw["primal_feasible"])
        self.assertTrue(result.primal_feasible)
        self.assertEqual(result.primal_upper_bound, result.objective)
        self.assertIsNotNone(result.raw["primal_safe_gap"])
        self.assertIsNotNone(result.primal_safe_gap)

    def test_api_never_labels_an_infeasible_point_as_an_upper_bound(
        self,
    ) -> None:
        problem = small_instance(dimension=12)
        infeasible = np.zeros(problem.dimension)
        infeasible[0] = 1.0

        with patch(
            "sksfolio.relaxation.api._backend_solver",
            return_value=lambda instance, options: {
                "status": "suboptimal",
                "x": infeasible,
                "has_solution": True,
                "effective_tolerance": 1e-8,
            },
        ):
            result = solve_relaxation(problem, "gurobi")

        self.assertIsNotNone(result.objective)
        self.assertFalse(result.raw["primal_feasible"])
        self.assertFalse(result.primal_feasible)
        self.assertIsNone(result.primal_upper_bound)
        self.assertIsNone(result.raw["primal_safe_gap"])
        self.assertIsNone(result.primal_safe_gap)
        self.assertIsNone(result.raw["bound_consistency_violation"])


class NativeDispatchTests(unittest.TestCase):
    def test_auto_uses_benchmark_policy_and_native_is_selectable(self) -> None:
        available = [
            backend
            for backend in ("fista", "pdhg", "scsdg")
            if native_available(backend)
        ]
        if not available:
            self.skipTest("compiled first-order extensions are not installed")

        problem = small_instance(dimension=12)
        for backend in available:
            with self.subTest(backend=backend, implementation="auto"):
                automatic = solve_relaxation(
                    problem,
                    backend,
                    implementation="auto",
                    options=_quick_options(backend),
                )
                self.assertEqual(automatic.raw["implementation"], "python")
                self.assertEqual(automatic.raw["language"], "python")
                self.assertFalse(automatic.raw["native_solver_core"])
                self.assertEqual(
                    automatic.raw["implementation_selection"],
                    "matched_benchmark_auto_policy",
                )

            with self.subTest(backend=backend, implementation="native"):
                compiled = solve_relaxation(
                    problem,
                    backend,
                    implementation="native",
                    options=_quick_options(backend),
                )
                self.assertEqual(compiled.raw["implementation"], "native")
                self.assertEqual(compiled.raw["language"], "cython")
                self.assertTrue(compiled.raw["native_solver_core"])
                self.assertEqual(compiled.state.implementation, "native")
                self.assertTrue(np.all(np.isfinite(compiled.weights)))

    def test_forced_native_never_silently_falls_back(self) -> None:
        problem = small_instance(dimension=12)
        real_import = importlib.import_module

        def import_without_native(name: str, *args, **kwargs):
            if name == "sksfolio.relaxation.fista._native_solver":
                raise ImportError("simulated missing extension")
            return real_import(name, *args, **kwargs)

        with patch(
            "sksfolio.relaxation.native.importlib.import_module",
            side_effect=import_without_native,
        ):
            with self.assertRaisesRegex(RuntimeError, "compiled fista"):
                solve_relaxation(
                    problem,
                    "fista",
                    implementation="native",
                    options=_quick_options("fista"),
                )


class EstimatorWarmStartTests(unittest.TestCase):
    def test_fit_accepts_an_explicit_prior_state(self) -> None:
        problem = small_instance(dimension=12)
        estimator = PerspectiveRelaxation(
            backend="pdhg",
            implementation="python",
            variant="fixed",
            solver_params=_quick_options("pdhg"),
        ).fit(problem)
        first_state = estimator.state_

        estimator.fit(problem, warm_start=first_state)

        self.assertTrue(estimator.result_.raw["warm_start_used"])
        self.assertIsNotNone(estimator.state_)
        self.assertIsNot(estimator.state_, first_state)

    def test_constructor_true_reuses_state_after_the_first_fit(self) -> None:
        problem = small_instance(dimension=12)
        estimator = PerspectiveRelaxation(
            backend="pdhg",
            implementation="python",
            variant="fixed",
            warm_start=True,
            solver_params=_quick_options("pdhg"),
        )

        estimator.fit(problem)
        self.assertFalse(estimator.result_.raw["warm_start_used"])
        first_state = estimator.state_
        estimator.fit(problem)

        self.assertTrue(estimator.result_.raw["warm_start_used"])
        self.assertIsNotNone(estimator.state_)
        self.assertIsNot(estimator.state_, first_state)


if __name__ == "__main__":
    unittest.main()
