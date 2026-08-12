"""End-to-end exactness tests for the certificate-driven BnB solver."""

from __future__ import annotations

import importlib.util
import math
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse

from sksfolio import MarkowitzInstance, solve_incumbent, solve_relaxation
from sksfolio.benchmarks.instance_generator import generate_instance
from sksfolio.bnb.solver import _Search, solve_bnb
from sksfolio.incumbent import (
    IncumbentResult,
    RestrictedQPResult,
    evaluate_incumbent,
)

from tests._bnb_reference import enumerate_sparse_supports
from tests.unit.test_safe_screening import screening_instance


def _pattern_optimum(enumerated, forced_one=(), forced_zero=()) -> float:
    required = set(int(index) for index in forced_one)
    forbidden = set(int(index) for index in forced_zero)
    return min(
        (
            solution.objective
            for solution in enumerated.solutions
            if required.issubset(solution.support)
            and forbidden.isdisjoint(solution.support)
        ),
        default=math.inf,
    )


def _solver_options(**updates):
    result = {
        "restricted_qp_options": {
            "feasibility_tolerance": 1e-8,
            "ftol": 1e-13,
            "max_iterations": 5000,
        }
    }
    result.update(updates)
    return result


class SmallExactBnBTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.problem = screening_instance()
        cls.enumerated = enumerate_sparse_supports(
            cls.problem,
            solver="scipy",
        )
        cls.relaxation = solve_relaxation(
            cls.problem,
            "fista",
            options={
                "max_iterations": 10_000,
                "check_interval": 10,
                "tolerance": 1e-10,
            },
        )
        cls.incumbent = solve_incumbent(
            cls.problem,
            cls.relaxation,
            method="auto",
            restricted_solver="scipy",
            random_state=3,
        )

    def test_root_solution_matches_exhaustive_enumeration(self) -> None:
        result = solve_bnb(
            self.problem,
            relaxation=self.relaxation,
            incumbent=self.incumbent,
            restricted_solver="scipy",
            options=_solver_options(),
        )
        self.assertEqual(result.status, "optimal")
        self.assertAlmostEqual(
            result.upper_bound,
            self.enumerated.objective,
            places=9,
        )
        self.assertLessEqual(
            result.lower_bound,
            self.enumerated.objective + 1e-10,
        )
        self.assertAlmostEqual(result.absolute_gap, 0.0, places=12)
        diagnostics = evaluate_incumbent(
            self.problem,
            result.weights,
            result.selectors,
        )
        self.assertTrue(diagnostics["numerically_feasible"])

    def test_required_and_forbidden_branches_match_enumeration(self) -> None:
        patterns = (
            ((0,), ()),
            ((), (0,)),
            ((0,), (7,)),
            ((7,), ()),
        )
        for required, forbidden in patterns:
            with self.subTest(required=required, forbidden=forbidden):
                incumbent = solve_incumbent(
                    self.problem,
                    self.relaxation,
                    method="auto",
                    restricted_solver="scipy",
                    random_state=3,
                    required_assets=required,
                    forbidden_assets=forbidden,
                )
                result = solve_bnb(
                    self.problem,
                    relaxation=self.relaxation,
                    incumbent=(incumbent if incumbent.feasible else None),
                    required_assets=required,
                    forbidden_assets=forbidden,
                    restricted_solver="scipy",
                    options=_solver_options(),
                )
                expected = _pattern_optimum(
                    self.enumerated,
                    required,
                    forbidden,
                )
                self.assertEqual(result.status, "optimal")
                self.assertAlmostEqual(result.upper_bound, expected, places=8)
                self.assertTrue(set(required).issubset(result.support))
                self.assertTrue(set(forbidden).isdisjoint(result.support))

    def test_osqp_certificate_can_complete_a_node_after_heuristic(self) -> None:
        if importlib.util.find_spec("osqp") is None:
            self.skipTest("OSQP is not installed")
        forbidden = (0,)
        incumbent = solve_incumbent(
            self.problem,
            self.relaxation,
            method="auto",
            restricted_solver="osqp",
            random_state=3,
            forbidden_assets=forbidden,
        )
        result = solve_bnb(
            self.problem,
            relaxation=self.relaxation,
            incumbent=incumbent,
            forbidden_assets=forbidden,
            restricted_solver="osqp",
            options={
                "restricted_qp_options": {
                    "eps_abs": 1e-10,
                    "eps_rel": 1e-10,
                    "feasibility_tolerance": 1e-8,
                }
            },
        )
        expected = _pattern_optimum(self.enumerated, (), forbidden)
        self.assertEqual(result.status, "optimal")
        self.assertAlmostEqual(result.upper_bound, expected, places=8)


class CutAndTerminationBnBTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.problem = generate_instance(
            dimension=12,
            rank=3,
            k=3,
            gamma_scale=100.0,
            regime="hybrid",
            seed=17,
            sectors=3,
            style_factors=2,
            stress_constraints=1,
            target_fraction=0.3,
            target_iterations=200,
            sector_band=0.25,
            style_band=0.75,
            stress_band=0.05,
            annual_volatility=0.20,
            common_correlation=0.15,
        )
        cls.enumerated = enumerate_sparse_supports(
            cls.problem,
            solver="scipy",
        )
        cls.relaxation = solve_relaxation(
            cls.problem,
            "fista",
            options={"max_iterations": 10_000, "tolerance": 1e-10},
        )
        cls.incumbent = solve_incumbent(
            cls.problem,
            cls.relaxation,
            method="auto",
            restricted_solver="scipy",
            random_state=17,
        )

    def test_multi_selector_cuts_preserve_the_exact_optimum(self) -> None:
        common = {
            "safe_screening": False,
            "maximum_cut_literals": 100,
        }
        without_cuts = solve_bnb(
            self.problem,
            relaxation=self.relaxation,
            incumbent=self.incumbent,
            restricted_solver="scipy",
            options=_solver_options(
                **common,
                multi_selector_cuts=False,
                root_pair_cuts=False,
            ),
        )
        with_cuts = solve_bnb(
            self.problem,
            relaxation=self.relaxation,
            incumbent=self.incumbent,
            restricted_solver="scipy",
            options=_solver_options(
                **common,
                multi_selector_cuts=True,
                root_pair_cuts=True,
            ),
        )
        for result in (without_cuts, with_cuts):
            self.assertEqual(result.status, "optimal")
            self.assertAlmostEqual(
                result.upper_bound,
                self.enumerated.objective,
                places=8,
            )
        self.assertAlmostEqual(
            with_cuts.upper_bound,
            without_cuts.upper_bound,
            places=11,
        )
        self.assertGreater(with_cuts.raw["root_pair_cuts_added"], 0)
        for cut in with_cuts.cuts:
            for solution in self.enumerated.solutions:
                selectors = solution.selectors
                matches = all(
                    selectors[index] > 0.5
                    for index in cut["forced_one"]
                ) and all(
                    selectors[index] < 0.5
                    for index in cut["forced_zero"]
                )
                if matches:
                    self.assertGreater(
                        solution.objective,
                        float(cut["upper_bound"]) + 1e-9,
                    )

    def test_node_limited_result_brackets_the_exact_optimum(self) -> None:
        result = solve_bnb(
            self.problem,
            relaxation=self.relaxation,
            incumbent=self.incumbent,
            restricted_solver="scipy",
            node_limit=1,
            options=_solver_options(
                safe_screening=False,
                multi_selector_cuts=False,
                root_pair_cuts=False,
            ),
        )
        self.assertIn(result.status, {"node_limit", "optimal"})
        self.assertLessEqual(
            result.lower_bound,
            self.enumerated.objective + 2e-8,
        )
        self.assertGreaterEqual(
            result.upper_bound,
            self.enumerated.objective - 2e-8,
        )

    def test_disabling_safe_screening_disables_node_fixings(self) -> None:
        search = object.__new__(_Search)
        search.settings = {"safe_screening": False}
        search.upper = 0.0
        mask = search._screen_mask(
            np.array([1, 3, 6], dtype=np.int64),
            np.array([math.inf, 1.0, 2.0]),
        )
        self.assertEqual(mask, 0)

    def test_sparse_infeasible_instance_is_proved_infeasible(self) -> None:
        dimension = 6
        matrix = np.vstack(
            [
                np.ones(dimension),
                np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0]),
            ]
        )
        problem = MarkowitzInstance(
            factor_loadings=np.arange(1.0, 13.0).reshape(6, 2) / 100.0,
            expected_returns=np.linspace(0.01, 0.03, dimension),
            constraint_matrix=sparse.csr_matrix(matrix),
            lower_bounds=np.array([1.0, 0.1, 0.1, 0.1]),
            upper_bounds=np.array([1.0, 1.0, 1.0, 1.0]),
            feasible_anchor=np.full(dimension, 1.0 / dimension),
            constraint_names=["budget", "group_1", "group_2", "group_3"],
            k=2,
            perspective_weight=0.2,
            return_reward=1.0,
        )
        problem.validate()
        relaxation = solve_relaxation(
            problem,
            "fista",
            options={"max_iterations": 10_000, "tolerance": 1e-9},
        )
        result = solve_bnb(
            problem,
            relaxation=relaxation,
            restricted_solver="scipy",
            node_limit=10_000,
            options=_solver_options(),
        )
        self.assertEqual(result.status, "infeasible")
        self.assertIsNone(result.upper_bound)


class BnBFailureAndTimeLimitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.problem = screening_instance()
        cls.relaxation = solve_relaxation(
            cls.problem,
            "fista",
            options={"max_iterations": 10_000, "tolerance": 1e-10},
        )
        cls.incumbent = solve_incumbent(
            cls.problem,
            cls.relaxation,
            method="auto",
            restricted_solver="scipy",
            random_state=3,
        )

    @staticmethod
    def _missing_incumbent() -> IncumbentResult:
        return IncumbentResult(
            {
                "status": "infeasible",
                "x": None,
                "selectors": None,
                "upper_bound": None,
                "numerically_feasible": False,
            }
        )

    def _mock_qp(self, certified: bool):
        def solve(instance, support, **kwargs):
            selected = np.asarray(support, dtype=np.int64).reshape(-1)
            return RestrictedQPResult(
                x=np.zeros(instance.dimension),
                support=selected,
                objective=None,
                feasible=False,
                status=("infeasible" if certified else "iteration_limit"),
                solver="mock",
                solve_seconds=0.0,
                raw_status=(
                    "primal infeasible"
                    if certified
                    else "maximum iterations reached"
                ),
                infeasibility_certified=certified,
            )

        return solve

    def test_uncertified_leaf_failure_is_not_an_infeasibility_proof(
        self,
    ) -> None:
        with patch(
            "sksfolio.bnb.solver.solve_incumbent",
            return_value=self._missing_incumbent(),
        ), patch(
            "sksfolio.bnb.solver.solve_restricted_qp",
            side_effect=self._mock_qp(False),
        ) as restricted:
            result = solve_bnb(
                self.problem,
                relaxation=self.relaxation,
                restricted_solver="scipy",
                node_limit=10_000,
                options=_solver_options(
                    safe_screening=False,
                    multi_selector_cuts=False,
                    root_pair_cuts=False,
                    node_dual_iterations=0,
                ),
            )
        self.assertEqual(result.status, "numerical_failure")
        self.assertGreater(result.raw["unresolved_leaves"], 0)
        self.assertGreater(restricted.call_count, 0)
        self.assertEqual(result.raw["restricted_qp_cache_hits"], 0)

    def test_certified_leaf_infeasibility_can_close_the_tree(self) -> None:
        with patch(
            "sksfolio.bnb.solver.solve_incumbent",
            return_value=self._missing_incumbent(),
        ), patch(
            "sksfolio.bnb.solver.solve_restricted_qp",
            side_effect=self._mock_qp(True),
        ) as restricted:
            result = solve_bnb(
                self.problem,
                relaxation=self.relaxation,
                restricted_solver="scipy",
                node_limit=10_000,
                options=_solver_options(
                    safe_screening=False,
                    multi_selector_cuts=False,
                    root_pair_cuts=False,
                    node_dual_iterations=0,
                ),
            )
        self.assertEqual(result.status, "infeasible")
        self.assertEqual(result.raw["unresolved_leaves"], 0)
        self.assertGreater(restricted.call_count, 0)

    def test_nested_limits_are_capped_by_the_global_limit(self) -> None:
        with patch(
            "sksfolio.bnb.solver.solve_relaxation",
            return_value=self.relaxation,
        ) as relaxation_solve, patch(
            "sksfolio.bnb.solver.solve_incumbent",
            return_value=self.incumbent,
        ) as incumbent_solve:
            solve_bnb(
                self.problem,
                time_limit=1.0,
                node_limit=1,
                restricted_solver="scipy",
                options={
                    "relaxation_options": {
                        "time_limit": 20.0,
                        "threads": 1,
                    },
                    "incumbent_options": {
                        "time_limit": 10.0,
                        "random_samples": 0,
                    },
                    "node_dual_iterations": 0,
                },
            )
        relaxation_options = relaxation_solve.call_args.kwargs["options"]
        incumbent_call = incumbent_solve.call_args.kwargs
        self.assertGreater(relaxation_options["time_limit"], 0.0)
        self.assertLessEqual(relaxation_options["time_limit"], 1.0)
        self.assertGreater(incumbent_call["time_limit"], 0.0)
        self.assertLessEqual(incumbent_call["time_limit"], 1.0)
        self.assertNotIn("time_limit", incumbent_call["options"])


if __name__ == "__main__":
    unittest.main()
