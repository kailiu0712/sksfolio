"""Julia bridge behavior that does not require a Julia installation."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

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


if __name__ == "__main__":
    unittest.main()
