"""Optional exact-objective agreement between sksfolio BnB and Gurobi."""

from __future__ import annotations

import importlib.util
import unittest

from sksfolio import solve_incumbent, solve_relaxation
from sksfolio.benchmarks.instance_generator import generate_instance
from sksfolio.bnb.solver import solve_bnb
from sksfolio.incumbent import solve_gurobi_incumbent


class BranchAndBoundGurobiTests(unittest.TestCase):
    def test_constrained_small_k_objective_agreement(self) -> None:
        if importlib.util.find_spec("gurobipy") is None:
            self.skipTest("gurobipy is not installed")
        problem = generate_instance(
            dimension=16,
            rank=3,
            k=4,
            gamma_scale=100.0,
            regime="hybrid",
            seed=29,
            sectors=4,
            style_factors=2,
            stress_constraints=2,
            target_fraction=0.3,
            target_iterations=200,
            sector_band=0.25,
            style_band=0.75,
            stress_band=0.05,
            annual_volatility=0.20,
            common_correlation=0.15,
        )
        relaxation = solve_relaxation(
            problem,
            "fista",
            options={"max_iterations": 10_000, "tolerance": 1e-10},
        )
        incumbent = solve_incumbent(
            problem,
            relaxation,
            method="auto",
            restricted_solver="scipy",
            random_state=29,
        )
        bnb = solve_bnb(
            problem,
            relaxation=relaxation,
            incumbent=incumbent,
            restricted_solver="scipy",
            time_limit=60.0,
            options={
                "restricted_qp_options": {
                    "feasibility_tolerance": 1e-8,
                    "ftol": 1e-13,
                }
            },
        )
        self.assertEqual(bnb.status, "optimal")

        gurobi = solve_gurobi_incumbent(
            problem,
            warm_start=bnb,
            time_limit=60.0,
            options={
                "verbose": False,
                "Threads": 1,
                "MIPFocus": 0,
                "MIPGap": 0.0,
                "MIPGapAbs": 1e-10,
                "FeasibilityTol": 1e-9,
                "OptimalityTol": 1e-9,
                "Seed": 0,
            },
        )
        if gurobi.status == "unavailable":
            self.skipTest(gurobi.raw.get("error", "Gurobi is unavailable"))
        if gurobi.status != "optimal":
            self.skipTest(
                "Gurobi did not complete an exact reference solve: "
                + gurobi.status
            )
        self.assertTrue(gurobi.feasible)
        self.assertAlmostEqual(
            bnb.upper_bound,
            gurobi.upper_bound,
            delta=2e-7 * max(1.0, abs(gurobi.upper_bound)),
        )
        self.assertLessEqual(
            bnb.lower_bound,
            gurobi.upper_bound + 2e-8,
        )
        self.assertLessEqual(
            gurobi.raw["solver_objective_bound"],
            bnb.upper_bound + 2e-8,
        )


if __name__ == "__main__":
    unittest.main()
