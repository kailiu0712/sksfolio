"""JuMP mixed-integer incumbent wrapper tests without launching Julia."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

import sksfolio
from sksfolio.incumbent import solve_jump_incumbent
from sksfolio.incumbent.jump import solve_jump_incumbent as direct_solve
from sksfolio.relaxation._julia_runner import JULIA_SOURCE

from tests._helpers import small_instance


def _unavailable(optimizer: str) -> dict:
    return {
        "status": "unavailable",
        "solver": optimizer,
        "formulation": "exact_binary_perspective_misocp",
        "x": None,
        "selectors": None,
        "upper_bound": None,
        "numerically_feasible": False,
    }


class JumpIncumbentTests(unittest.TestCase):
    def test_solver_is_exported_from_both_public_namespaces(self) -> None:
        self.assertIs(sksfolio.solve_jump_incumbent, direct_solve)
        self.assertIs(solve_jump_incumbent, direct_solve)

    def test_default_optimizer_is_gurobi(self) -> None:
        problem = small_instance(dimension=8)
        with patch(
            "sksfolio.incumbent.jump.solve_binary_julia",
            return_value=_unavailable("gurobi"),
        ) as bridge:
            result = solve_jump_incumbent(problem)
        self.assertEqual(result.status, "unavailable")
        self.assertEqual(bridge.call_args.args[1], "gurobi")

    def test_optimizer_aliases_and_custom_constructor_are_forwarded(
        self,
    ) -> None:
        problem = small_instance(dimension=8)
        for optimizer in ("gurobi", "mosek", "SCIP.Optimizer"):
            with self.subTest(optimizer=optimizer), patch(
                "sksfolio.incumbent.jump.solve_binary_julia",
                return_value=_unavailable(optimizer),
            ) as bridge:
                result = solve_jump_incumbent(
                    problem,
                    optimizer=optimizer,
                    options={"threads": 2, "relative_gap": 1e-5},
                )
            self.assertEqual(result.status, "unavailable")
            called_problem, called_optimizer, called_options = (
                bridge.call_args.args
            )
            self.assertIs(called_problem, problem)
            self.assertEqual(called_optimizer, optimizer)
            self.assertEqual(called_options["threads"], 2)
            self.assertEqual(called_options["relative_gap"], 1e-5)

    def test_branch_sets_are_validated_and_forwarded_by_keyword(self) -> None:
        problem = small_instance(dimension=8)
        with patch(
            "sksfolio.incumbent.jump.solve_binary_julia",
            return_value=_unavailable("mosek"),
        ) as bridge:
            solve_jump_incumbent(
                problem,
                optimizer="mosek",
                required_assets=(1, 3),
                forbidden_assets=(5, 7),
                time_limit=12.5,
                options={"threads": 1},
            )
        called_problem, called_optimizer, called_options = (
            bridge.call_args.args
        )
        self.assertIs(called_problem, problem)
        self.assertEqual(called_optimizer, "mosek")
        self.assertEqual(called_options["threads"], 1)
        self.assertEqual(called_options["time_limit"], 12.5)
        np.testing.assert_array_equal(
            bridge.call_args.kwargs["required_assets"],
            np.array([1, 3]),
        )
        np.testing.assert_array_equal(
            bridge.call_args.kwargs["forbidden_assets"],
            np.array([5, 7]),
        )

    def test_julia_source_contains_exact_binary_perspective_model(
        self,
    ) -> None:
        source = JULIA_SOURCE.read_text(encoding="utf-8")
        self.assertIn("export solve_binary_bundle", source)
        self.assertIn("function solve_binary_bundle(", source)
        binary_model = source.split(
            "function _build_binary_model(",
            maxsplit=1,
        )[1].split(
            "function _binary_solver_snapshot",
            maxsplit=1,
        )[0]
        self.assertIn("options.required_assets", binary_model)
        self.assertIn("options.forbidden_assets", binary_model)
        self.assertRegex(
            binary_model,
            r"@variable\(model,\s*z\[1:dimension\].*(Bin|binary)",
        )
        self.assertIn(
            "[0.5 * t[index]; z[index]; x[index]] in",
            binary_model,
        )
        self.assertIn("RotatedSecondOrderCone()", binary_model)
        self.assertIn("values = instance.C * x", binary_model)
        self.assertIn("isfinite(lower)", binary_model)
        self.assertIn("@constraint(model, values[row] >= lower)", binary_model)
        self.assertIn("isfinite(upper)", binary_model)
        self.assertIn("@constraint(model, values[row] <= upper)", binary_model)


if __name__ == "__main__":
    unittest.main()
