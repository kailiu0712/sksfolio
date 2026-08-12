"""Generic Julia/JuMP implementation of the perspective relaxation."""

from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Optional

from .._julia_runner import solve_julia


DEFAULT_OPTIMIZER = "clarabel"
OPTIMIZER_ALIASES = (
    "gurobi",
    "mosek",
    "mosektools",
    "clarabel",
    "cosmo",
    "highs",
)
_JULIA_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _optimizer_specification(options: Dict[str, Any]) -> str:
    """Remove and normalize the JuMP optimizer selector."""
    optimizer = options.pop("optimizer", None)
    solver = options.pop("solver", None)
    package = options.pop("optimizer_package", None)
    constructor = options.pop("optimizer_name", None)
    if optimizer is not None and solver is not None:
        raise ValueError(
            "use either options['optimizer'] or options['solver'], not both"
        )
    if optimizer is None:
        optimizer = solver
    if optimizer is not None and (
        package is not None or constructor is not None
    ):
        raise ValueError(
            "use either options['optimizer'] or optimizer_package/"
            "optimizer_name, not both"
        )
    if optimizer is not None:
        specification = str(optimizer).strip()
        if not specification:
            raise ValueError("optimizer must not be empty")
        return specification
    if package is None and constructor is None:
        return DEFAULT_OPTIMIZER
    if package is None:
        raise ValueError(
            "optimizer_package is required when optimizer_name is set"
        )
    package_name = str(package).strip()
    constructor_name = (
        "Optimizer" if constructor is None else str(constructor).strip()
    )
    if not _JULIA_IDENTIFIER.fullmatch(package_name):
        raise ValueError("optimizer_package must be a Julia identifier")
    if not _JULIA_IDENTIFIER.fullmatch(constructor_name):
        raise ValueError("optimizer_name must be a Julia identifier")
    return f"{package_name}.{constructor_name}"


def solve(
    problem: Any,
    options: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Solve through JuMP with a selected optimizer.

    Use ``options={"optimizer": "clarabel"}`` for a built-in alias, or
    ``optimizer_package`` and ``optimizer_name`` for another installed
    MathOptInterface optimizer. The exact formulation requires rotated
    second-order-cone support; consequently HiGHS is recognized but
    rejected with a structured ``unsupported`` result.
    """
    supplied = dict(options or {})
    optimizer = _optimizer_specification(supplied)
    normalized = optimizer.lower()
    if normalized in {
        "clarabel",
        "clarabel.optimizer",
        "cosmo",
        "cosmo.optimizer",
    }:
        supplied.setdefault("julia_instantiate", True)
    return solve_julia(problem, optimizer, supplied)


__all__ = [
    "DEFAULT_OPTIMIZER",
    "OPTIMIZER_ALIASES",
    "solve",
]
