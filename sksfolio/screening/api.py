"""Convenience entry point for constraint-aware safe screening."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence, Union

from ..relaxation.problem import MarkowitzInstance, load_instance_bundle
from .oracle import FenchelScreeningOracle
from .result import SafeScreeningResult


ProblemLike = Union[MarkowitzInstance, str, Path]


def safe_screen(
    problem: ProblemLike,
    relaxation: Any,
    incumbent: Any,
    *,
    forced_one: Sequence[int] = (),
    forced_zero: Sequence[int] = (),
    propagate: bool = True,
    absolute_margin: float = 0.0,
    relative_margin: float = 1e-10,
    feasibility_tolerance: float = 1e-7,
) -> SafeScreeningResult:
    """Screen binary selectors using a safe dual and feasible incumbent."""
    instance = (
        problem
        if isinstance(problem, MarkowitzInstance)
        else load_instance_bundle(problem)
    )
    oracle = FenchelScreeningOracle(instance, relaxation)
    return oracle.screen(
        incumbent,
        forced_one=forced_one,
        forced_zero=forced_zero,
        propagate=propagate,
        absolute_margin=absolute_margin,
        relative_margin=relative_margin,
        feasibility_tolerance=feasibility_tolerance,
    )


__all__ = ["safe_screen"]
