"""Portable restart-state tests for the corrected algorithms."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from sksfolio.relaxation import (
    CORRECTED_ALGORITHMS,
    RelaxationState,
    solve_relaxation,
)

from tests._helpers import small_instance


class RelaxationStateTests(unittest.TestCase):
    def test_npz_round_trip_preserves_fields(self) -> None:
        state = RelaxationState(
            backend="corrected_lbfgs",
            implementation="python",
            dimension=4,
            k=2,
            constraint_ids=("budget", "sector"),
            x=np.array([0.1, 0.2, 0.3, 0.4]),
            factor_dual=np.array([0.25, -0.75]),
            constraint_dual=np.array([3.0, -2.0]),
            arrays={"momentum_x": np.arange(4, dtype=float)},
            scalars={"lipschitz": 7.5},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = state.save(Path(directory) / "state.npz")
            restored = RelaxationState.load(path)
        self.assertEqual(restored.backend, state.backend)
        self.assertEqual(restored.constraint_ids, state.constraint_ids)
        self.assertEqual(restored.scalars, state.scalars)
        np.testing.assert_array_equal(restored.x, state.x)
        np.testing.assert_array_equal(restored.factor_dual, state.factor_dual)
        np.testing.assert_array_equal(
            restored.constraint_dual, state.constraint_dual
        )
        np.testing.assert_array_equal(
            restored.arrays["momentum_x"], state.arrays["momentum_x"]
        )

    def test_constraint_duals_are_mapped_by_name(self) -> None:
        state = RelaxationState(
            backend="corrected_fista",
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

    def test_both_algorithms_accept_prior_results(self) -> None:
        problem = small_instance(dimension=12)
        for algorithm in CORRECTED_ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                options = {
                    "threads": 1,
                    "max_iterations": 100,
                    "tolerance": 1e-5,
                    "prox_max_iterations": 1_000,
                }
                first = solve_relaxation(problem, algorithm, options=options)
                restarted = solve_relaxation(
                    problem,
                    algorithm,
                    warm_start=first,
                    options=options,
                )
                self.assertTrue(restarted.raw["warm_start_used"])
                self.assertTrue(restarted.raw["warm_start_exact_structure"])
                self.assertEqual(restarted.state.backend, algorithm)
                self.assertTrue(restarted.dual_certificate.verify(problem))


if __name__ == "__main__":
    unittest.main()
