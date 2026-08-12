"""Julia bridge behavior that does not require a Julia installation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from sksfolio.relaxation import _julia_runner as runner
from sksfolio.relaxation._julia_runner import _load_module


class JuliaRunnerTests(unittest.TestCase):
    def test_loaded_module_is_reused_without_replacement(self) -> None:
        julia = Mock()
        julia.seval.side_effect = [None, True]
        julia.MarkowitzCommercial = object()
        package = SimpleNamespace(Main=julia)
        with patch(
            "sksfolio.relaxation._julia_runner.importlib.import_module",
            return_value=package,
        ):
            module = _load_module(False)
        self.assertIs(module, julia.MarkowitzCommercial)
        julia.include.assert_not_called()

    def test_instantiation_resolves_the_extended_project(self) -> None:
        julia = Mock()
        julia.seval.side_effect = [None, True]
        julia.MarkowitzCommercial = object()
        packages = {
            "juliapkg": SimpleNamespace(
                executable=lambda: "/fake/bin/julia"
            ),
            "juliacall": SimpleNamespace(Main=julia),
        }
        with (
            patch.object(
                runner.importlib,
                "import_module",
                side_effect=lambda name: packages[name],
            ),
            patch.object(runner.subprocess, "run") as run,
        ):
            _load_module(True)
        command = run.call_args.args[0]
        self.assertIn("Pkg.resolve()", command[-1])
        self.assertIn("Pkg.instantiate()", command[-1])
        self.assertTrue(run.call_args.kwargs["check"])

    def test_binary_bridge_forwards_branch_sets_and_converts_vectors(
        self,
    ) -> None:
        solve_binary_bundle = Mock(
            return_value={
                "status": "optimal",
                "x": [0.0, 1.0, 0.0],
                "selectors": [0.0, 1.0, 0.0],
            }
        )
        module = SimpleNamespace(
            solve_binary_bundle=solve_binary_bundle,
        )
        with patch.object(runner, "_load_module", return_value=module):
            result = runner._solve_binary_bundle(
                Path("/tmp/example-bundle"),
                "mosek",
                {"threads": 1},
                np.array([1], dtype=np.int64),
                np.array([0, 2], dtype=np.int64),
                False,
            )
        solve_binary_bundle.assert_called_once_with(
            "/tmp/example-bundle",
            "mosek",
            {"threads": 1},
            [1],
            [0, 2],
        )
        np.testing.assert_array_equal(result["x"], [0.0, 1.0, 0.0])
        np.testing.assert_array_equal(
            result["selectors"],
            [0.0, 1.0, 0.0],
        )


if __name__ == "__main__":
    unittest.main()
