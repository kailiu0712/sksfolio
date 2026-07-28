"""Julia/JuMP Gurobi implementation of the perspective relaxation."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from .._julia_runner import solve_julia


def solve(
    problem: Any,
    options: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Solve through JuMP with Gurobi."""
    return solve_julia(problem, "gurobi", options)


__all__ = ["solve"]
