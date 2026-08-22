"""Corrected outer FISTA and inner proximal-oracle tests."""

from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from sksfolio.relaxation import solve_relaxation
from sksfolio.relaxation.fista import LinearConstraintProx

from tests._helpers import small_instance


def _budget_only_instance():
    problem = small_instance()
    result = replace(
        problem,
        constraint_matrix=problem.C[:1, :].copy(),
        lower_bounds=np.array([1.0]),
        upper_bounds=np.array([1.0]),
        constraint_names=["budget"],
    )
    result.validate()
    return result


class CorrectedFISTATests(unittest.TestCase):
    def test_public_algorithms_select_the_expected_inner_solver(self) -> None:
        problem = small_instance()
        expected = {
            "corrected_fista": "row_scaled_dual_fista",
            "corrected_lbfgs": "row_scaled_dual_lbfgsb",
        }
        for algorithm, method in expected.items():
            with self.subTest(algorithm=algorithm):
                result = solve_relaxation(
                    problem,
                    algorithm,
                    options={
                        "threads": 1,
                        "tolerance": 1e-5,
                        "feasibility_tolerance": 1e-5,
                        "max_iterations": 2_000,
                        "prox_tolerance": 1e-7,
                        "prox_max_iterations": 2_000,
                    },
                )
                self.assertEqual(result.status, "converged")
                self.assertEqual(result.raw["prox_oracle_used"], method)
                self.assertEqual(result.raw["algorithm_variant"], "corrected")
                self.assertEqual(result.raw["momentum_rule"], "classical_fista")
                self.assertTrue(result.dual_certificate.verify(problem))

    def test_budget_fast_path_supports_both_pava_methods(self) -> None:
        problem = _budget_only_instance()
        values = {}
        for method in ("full_sort", "partial_sort"):
            result = solve_relaxation(
                problem,
                "corrected_lbfgs",
                pava=method,
                options={"threads": 1, "tolerance": 1e-7, "max_iterations": 2_000},
            )
            values[method] = result.objective
            self.assertEqual(result.status, "converged")
            self.assertEqual(
                result.raw["prox_oracle_used"],
                "exact_budget_scalar_brent",
            )
            self.assertTrue(result.raw["prox_exact_budget_fast_path"])
            self.assertEqual(result.raw["pava_backend"], method)
        self.assertAlmostEqual(values["full_sort"], values["partial_sort"], places=7)

    def test_inner_solvers_agree_and_warm_start(self) -> None:
        problem = small_instance()
        rng = np.random.default_rng(7)
        argument = rng.normal(size=problem.dimension)
        common = dict(
            pava_method="partial_sort",
            tolerance=1e-8,
            max_iterations=2_000,
            use_budget_fast_path=False,
        )
        fista = LinearConstraintProx(
            problem.C, problem.lower, problem.upper, problem.k,
            dual_solver="fista", **common
        )
        lbfgs = LinearConstraintProx(
            problem.C, problem.lower, problem.upper, problem.k,
            dual_solver="lbfgs", lbfgs_fallback=False, **common
        )
        first = fista.solve(argument, 0.7)
        second = fista.solve(argument + 1e-5, 0.7)
        quasi_newton = lbfgs.solve(argument, 0.7)
        self.assertTrue(first.converged)
        self.assertTrue(second.warm_started)
        self.assertTrue(quasi_newton.converged)
        np.testing.assert_allclose(first.x, quasi_newton.x, atol=3e-7)


if __name__ == "__main__":
    unittest.main()
