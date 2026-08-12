"""Optional exact-objective agreement for the direct MOSEK MISOCP backend."""

from __future__ import annotations

import importlib.util
import unittest

from sksfolio.incumbent import solve_mosek_incumbent

from tests._bnb_reference import enumerate_sparse_supports
from tests.unit.test_safe_screening import screening_instance


class BranchAndBoundMosekTests(unittest.TestCase):
    def test_constrained_small_k_objective_agreement(self) -> None:
        if importlib.util.find_spec("mosek") is None:
            self.skipTest("MOSEK is not installed")
        problem = screening_instance()
        required = (0,)
        forbidden = (7,)
        reference = enumerate_sparse_supports(
            problem,
            required_assets=required,
            forbidden_assets=forbidden,
            solver="scipy",
        )
        mosek = solve_mosek_incumbent(
            problem,
            required_assets=required,
            forbidden_assets=forbidden,
            time_limit=60.0,
            options={
                "threads": 1,
                "relative_gap": 0.0,
                "absolute_gap": 1e-10,
                "log": False,
            },
        )
        if mosek.status == "unavailable":
            self.skipTest(mosek.raw.get("error", "MOSEK is unavailable"))
        self.assertNotEqual(
            mosek.status,
            "error",
            msg=mosek.raw.get("error"),
        )
        if mosek.status != "optimal":
            self.skipTest(
                "MOSEK did not complete an exact reference solve: "
                + mosek.status
            )
        self.assertTrue(mosek.feasible)
        self.assertTrue(set(required).issubset(mosek.support))
        self.assertTrue(set(forbidden).isdisjoint(mosek.support))
        self.assertAlmostEqual(
            mosek.upper_bound,
            reference.objective,
            delta=2e-7 * max(1.0, abs(reference.objective)),
        )
        solver_bound = mosek.raw.get("solver_objective_bound")
        if solver_bound is not None:
            self.assertLessEqual(
                float(solver_bound),
                reference.objective + 2e-7,
            )


if __name__ == "__main__":
    unittest.main()
