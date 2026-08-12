"""Agreement checks for optional native commercial backends."""

from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from sksfolio.relaxation import solve_relaxation
from sksfolio.relaxation.fista import prox_budget
from sksfolio.relaxation.pdhg.pava import (
    prox_full_sort,
    prox_partial_sort,
)

from tests._helpers import small_instance


class CommercialBackendTests(unittest.TestCase):
    def test_budget_prox_matches_an_independent_conic_model(
        self,
    ) -> None:
        if importlib.util.find_spec("gurobipy") is None:
            self.skipTest("gurobipy is not installed")
        import gurobipy as gp

        values = np.array([1.9, -0.8, 0.6, 2.5, 0.1, 1.2, -0.3])
        gamma = 0.85
        k = 3
        environment = gp.Env(empty=True)
        environment.setParam("OutputFlag", 0)
        try:
            environment.start()
        except Exception as error:
            environment.dispose()
            self.skipTest(f"Gurobi environment is unusable: {error}")
        model = None
        try:
            model = gp.Model("budget_prox_validation", env=environment)
            model.Params.OutputFlag = 0
            model.Params.NonConvex = 0
            model.Params.Method = 2
            model.Params.Crossover = 0
            model.Params.FeasibilityTol = 1e-9
            model.Params.OptimalityTol = 1e-9
            model.Params.BarConvTol = 1e-10
            model.Params.BarQCPConvTol = 1e-10
            x = model.addMVar(values.size, lb=0.0, ub=1.0)
            z = model.addMVar(values.size, lb=0.0, ub=1.0)
            t = model.addMVar(values.size, lb=0.0)
            model.addConstr(x <= z)
            model.addConstr(z.sum() <= k)
            model.addConstr(x * x <= t * z)
            model.addConstr(x.sum() == 1.0)
            difference = x - values
            model.setObjective(
                0.5 * (difference @ difference)
                + 0.5 * gamma * t.sum(),
                gp.GRB.MINIMIZE,
            )
            model.optimize()
            if model.Status != gp.GRB.OPTIMAL:
                self.skipTest("Gurobi conic validation did not solve")
            expected = np.asarray(x.X, dtype=float)
        finally:
            if model is not None:
                model.dispose()
            environment.dispose()
        for method in ("full_sort", "partial_sort"):
            np.testing.assert_allclose(
                prox_budget(values, gamma, k, method),
                expected,
                atol=2e-6,
            )

    def test_both_pava_oracles_match_an_independent_conic_model(
        self,
    ) -> None:
        if importlib.util.find_spec("gurobipy") is None:
            self.skipTest("gurobipy is not installed")
        import gurobipy as gp

        values = np.array([2.4, -0.5, 1.1, 4.0, 0.2, 1.8])
        gamma = 0.7
        k = 3
        environment = gp.Env(empty=True)
        environment.setParam("OutputFlag", 0)
        try:
            environment.start()
        except Exception as error:
            environment.dispose()
            self.skipTest(f"Gurobi environment is unusable: {error}")
        model = None
        try:
            model = gp.Model("pava_validation", env=environment)
            model.Params.OutputFlag = 0
            model.Params.NonConvex = 0
            model.Params.Method = 2
            model.Params.Crossover = 0
            model.Params.FeasibilityTol = 1e-9
            model.Params.OptimalityTol = 1e-9
            model.Params.BarConvTol = 1e-10
            model.Params.BarQCPConvTol = 1e-10
            x = model.addMVar(values.size, lb=0.0, ub=1.0)
            z = model.addMVar(values.size, lb=0.0, ub=1.0)
            t = model.addMVar(values.size, lb=0.0)
            model.addConstr(x <= z)
            model.addConstr(z.sum() <= k)
            model.addConstr(x * x <= t * z)
            difference = x - values
            model.setObjective(
                0.5 * (difference @ difference)
                + 0.5 * gamma * t.sum(),
                gp.GRB.MINIMIZE,
            )
            model.optimize()
            if model.Status != gp.GRB.OPTIMAL:
                self.skipTest("Gurobi conic validation did not solve")
            expected = np.asarray(x.X, dtype=float)
        finally:
            if model is not None:
                model.dispose()
            environment.dispose()
        np.testing.assert_allclose(
            prox_full_sort(values, gamma, k),
            expected,
            atol=2e-6,
        )
        np.testing.assert_allclose(
            prox_partial_sort(values, gamma, k),
            expected,
            atol=2e-6,
        )

    def test_native_gurobi_agrees_with_pdhg_dual_bound(self) -> None:
        if importlib.util.find_spec("gurobipy") is None:
            self.skipTest("gurobipy is not installed")
        problem = small_instance()
        gurobi = solve_relaxation(
            problem,
            "gurobi.python",
            options={
                "threads": 1,
                "tolerance": 1e-9,
                "time_limit": 30.0,
                "log": False,
            },
        )
        if not gurobi.raw.get("has_solution", False):
            self.skipTest(
                "Gurobi is installed but no licensed solution is available"
            )
        self.assertIsNotNone(gurobi.solver_objective_bound)
        self.assertIsNotNone(gurobi.safe_dual_bound)
        self.assertTrue(gurobi.dual_certificate.verify(problem))
        self.assertLessEqual(
            gurobi.safe_dual_bound,
            gurobi.objective + 1e-8,
        )
        self.assertNotEqual(
            gurobi.raw["dual_bound_source"],
            gurobi.raw["solver_objective_bound_source"],
        )
        pdhg = solve_relaxation(
            problem,
            "pdhg",
            variant="metric-linesearch-restart",
            pava="partial_sort",
            options={
                "threads": 1,
                "tolerance": 1e-7,
                "feasibility_tolerance": 1e-7,
                "max_iterations": 10000,
                "check_interval": 20,
                "min_epoch": 20,
                "max_epoch": 500,
            },
        )
        self.assertTrue(pdhg.dual_certificate.verify(problem))
        self.assertEqual(pdhg.status, "converged")
        self.assertLessEqual(
            pdhg.raw["diagnostics"]["violations"]["maximum"],
            1e-7,
        )
        self.assertLessEqual(
            pdhg.safe_dual_bound,
            gurobi.objective + 1e-8,
        )
        relative_dual_error = (
            gurobi.objective - pdhg.safe_dual_bound
        ) / max(1.0, abs(gurobi.objective))
        self.assertLess(relative_dual_error, 2e-4)
        self.assertAlmostEqual(
            pdhg.objective,
            gurobi.objective,
            delta=2e-7 * max(1.0, abs(gurobi.objective)),
        )

    def test_native_mosek_is_optional(self) -> None:
        if importlib.util.find_spec("mosek") is None:
            self.skipTest("MOSEK is not installed")
        result = solve_relaxation(
            small_instance(),
            "mosek.python",
            options={
                "threads": 1,
                "tolerance": 1e-8,
                "time_limit": 30.0,
                "log": False,
            },
        )
        if not result.raw.get("has_solution", False):
            self.skipTest("MOSEK license is unavailable")
        self.assertIsNotNone(result.objective)
        if importlib.util.find_spec("gurobipy") is not None:
            gurobi = solve_relaxation(
                small_instance(),
                "gurobi.python",
                options={
                    "threads": 1,
                    "tolerance": 1e-9,
                    "time_limit": 30.0,
                    "log": False,
                },
            )
            if gurobi.raw.get("has_solution", False):
                self.assertAlmostEqual(
                    result.objective,
                    gurobi.objective,
                    delta=2e-7 * max(1.0, abs(gurobi.objective)),
                )


if __name__ == "__main__":
    unittest.main()
