"""Generic JuMP wrapper tests with no Julia installation required."""

from __future__ import annotations

import tomllib
import unittest
from unittest.mock import patch

import numpy as np

from sksfolio.relaxation._julia_runner import JULIA_DIRECTORY, JULIA_SOURCE
from sksfolio.relaxation.api import solve_relaxation
from sksfolio.relaxation.jump.julia import (
    DEFAULT_OPTIMIZER,
    _optimizer_specification,
    solve,
)
from sksfolio.relaxation.gurobi.julia import solve as solve_gurobi_julia
from sksfolio.relaxation.mosek.julia import solve as solve_mosek_julia
from sksfolio.relaxation.state import RelaxationState

from tests._helpers import small_instance


class JumpBackendTests(unittest.TestCase):
    def test_default_optimizer_is_clarabel(self) -> None:
        options = {}
        self.assertEqual(_optimizer_specification(options), "clarabel")
        self.assertEqual(DEFAULT_OPTIMIZER, "clarabel")
        self.assertEqual(options, {})

    def test_builtin_optimizer_is_removed_before_julia_call(self) -> None:
        instance = small_instance()
        with patch(
            "sksfolio.relaxation.jump.julia.solve_julia",
            return_value={"status": "unavailable"},
        ) as bridge:
            result = solve(
                instance,
                {
                    "optimizer": "cosmo",
                    "tolerance": 1e-5,
                    "optimizer_attributes": {"max_iter": 100},
                },
            )
        self.assertEqual(result["status"], "unavailable")
        bridge.assert_called_once()
        problem, optimizer, options = bridge.call_args.args
        self.assertIs(problem, instance)
        self.assertEqual(optimizer, "cosmo")
        self.assertEqual(
            options,
            {
                "tolerance": 1e-5,
                "optimizer_attributes": {"max_iter": 100},
                "julia_instantiate": True,
            },
        )

    def test_custom_package_and_constructor_are_supported(self) -> None:
        options = {
            "optimizer_package": "SCS",
            "optimizer_name": "Optimizer",
            "log": True,
        }
        self.assertEqual(
            _optimizer_specification(options),
            "SCS.Optimizer",
        )
        self.assertEqual(options, {"log": True})

    def test_custom_package_defaults_to_optimizer_constructor(self) -> None:
        options = {"optimizer_package": "ECOS"}
        self.assertEqual(
            _optimizer_specification(options),
            "ECOS.Optimizer",
        )

    def test_solver_is_an_alias_for_optimizer(self) -> None:
        options = {"solver": "gurobi", "threads": 1}
        self.assertEqual(_optimizer_specification(options), "gurobi")
        self.assertEqual(options, {"threads": 1})

    def test_conflicting_optimizer_selectors_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "either"):
            _optimizer_specification(
                {
                    "optimizer": "clarabel",
                    "optimizer_package": "COSMO",
                }
            )

    def test_invalid_custom_identifier_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Julia identifier"):
            _optimizer_specification(
                {"optimizer_package": "Unsafe; import Bad"}
            )

    def test_legacy_julia_aliases_keep_their_optimizer(self) -> None:
        instance = small_instance()
        cases = (
            (
                "sksfolio.relaxation.gurobi.julia.solve_julia",
                solve_gurobi_julia,
                "gurobi",
            ),
            (
                "sksfolio.relaxation.mosek.julia.solve_julia",
                solve_mosek_julia,
                "mosek",
            ),
        )
        for target, adapter, expected in cases:
            with self.subTest(optimizer=expected), patch(
                target,
                return_value={"status": "unavailable"},
            ) as bridge:
                adapter(instance, {"threads": 1})
            bridge.assert_called_once_with(
                instance,
                expected,
                {"threads": 1},
            )

    def test_bundled_project_declares_generic_optimizers(self) -> None:
        with (JULIA_DIRECTORY / "Project.toml").open("rb") as stream:
            project = tomllib.load(stream)
        for package in (
            "Gurobi",
            "MosekTools",
            "Clarabel",
            "COSMO",
            "HiGHS",
        ):
            self.assertIn(package, project["deps"])

    def test_julia_source_guards_the_highs_formulation(self) -> None:
        source = JULIA_SOURCE.read_text(encoding="utf-8")
        for alias in (
            '"gurobi"',
            '"mosektools"',
            '"clarabel"',
            '"cosmo"',
            '"highs"',
        ):
            self.assertIn(alias, source)
        self.assertIn("unsupported_formulation", source)
        self.assertIn("rotated_second_order_cone", source)

    def test_julia_source_accepts_restart_primal_state(self) -> None:
        source = JULIA_SOURCE.read_text(encoding="utf-8")
        self.assertIn(
            "initial_x::Union{Nothing, Vector{Float64}}",
            source,
        )
        self.assertIn("_validate_initial_x(instance, options)", source)
        self.assertIn('"initial_x_supplied"', source)
        self.assertIn('source_name = initial_x === nothing ?', source)

    def test_public_jump_warm_start_routes_primal_separately(self) -> None:
        instance = small_instance()
        state = RelaxationState(
            backend="fista",
            implementation="python",
            dimension=instance.dimension,
            k=instance.k,
            constraint_ids=tuple(instance.constraint_names),
            # Intentionally violates the budget equality row. A BnB
            # restart point need only remain in the perspective domain.
            x=np.zeros(instance.dimension),
        )
        captured = {}

        def fake_solver(problem, options):
            captured.update(options)
            return {
                "status": "unavailable",
                "has_solution": False,
                "x": None,
            }

        with patch(
            "sksfolio.relaxation.api._backend_solver",
            return_value=fake_solver,
        ):
            solve_relaxation(instance, "jump", warm_start=state)
        self.assertIs(captured["warm_start"], True)
        np.testing.assert_array_equal(
            captured["initial_x"],
            state.x,
        )


if __name__ == "__main__":
    unittest.main()
