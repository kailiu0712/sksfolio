"""Solver-independent exact references for small branch-and-bound tests.

The enumerated set is the binary selector support, not merely the nonzero
support of the portfolio weights.  A selected asset may optimally receive
zero weight, so preserving this distinction is necessary when validating
required/forbidden branches and no-good cuts.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from sksfolio.incumbent import solve_restricted_qp
from sksfolio.incumbent.support import validate_branch_indices
from sksfolio.relaxation import MarkowitzInstance


@dataclass(frozen=True)
class EnumeratedSupportSolution:
    objective: float
    support: tuple[int, ...]
    weights: np.ndarray

    @property
    def selectors(self) -> np.ndarray:
        values = np.zeros(self.weights.size)
        values[list(self.support)] = 1.0
        return values


@dataclass(frozen=True)
class ExhaustiveSparseResult:
    objective: float
    weights: np.ndarray
    selectors: np.ndarray
    support: tuple[int, ...]
    attempted_supports: int
    feasible_supports: int
    solutions: tuple[EnumeratedSupportSolution, ...]


def enumerate_sparse_supports(
    instance: MarkowitzInstance,
    *,
    required_assets: Sequence[int] = (),
    forbidden_assets: Sequence[int] = (),
    solver: str = "scipy",
    options: Optional[Mapping[str, Any]] = None,
) -> ExhaustiveSparseResult:
    """Solve every selector support of cardinality at most ``k``.

    This helper is intentionally limited to tiny correctness instances.  It
    uses the production fixed-support convex QP solver, but it does not use
    any branch-and-bound implementation or screening logic.
    """
    instance.validate()
    required, forbidden = validate_branch_indices(
        instance.dimension,
        instance.k,
        required_assets,
        forbidden_assets,
    )
    excluded = np.zeros(instance.dimension, dtype=bool)
    excluded[required] = True
    excluded[forbidden] = True
    free = np.flatnonzero(~excluded)
    settings = {
        "feasibility_tolerance": 1e-8,
        "ftol": 1e-13,
        "eps_abs": 1e-10,
        "eps_rel": 1e-10,
        "max_iterations": 50_000,
        "polishing": True,
    }
    settings.update(dict(options or {}))

    attempted = 0
    feasible: list[EnumeratedSupportSolution] = []
    required_tuple = tuple(int(index) for index in required)
    remaining_capacity = int(instance.k - required.size)
    for additional_size in range(remaining_capacity + 1):
        for additional in itertools.combinations(free, additional_size):
            support = tuple(
                sorted(
                    required_tuple
                    + tuple(int(index) for index in additional)
                )
            )
            attempted += 1
            result = solve_restricted_qp(
                instance,
                support,
                solver=solver,
                options=settings,
            )
            if result.feasible and result.objective is not None:
                feasible.append(
                    EnumeratedSupportSolution(
                        objective=float(result.objective),
                        support=support,
                        weights=np.asarray(result.x, dtype=float).copy(),
                    )
                )

    if not feasible:
        return ExhaustiveSparseResult(
            objective=math.inf,
            weights=np.zeros(instance.dimension),
            selectors=np.zeros(instance.dimension),
            support=(),
            attempted_supports=attempted,
            feasible_supports=0,
            solutions=(),
        )
    feasible.sort(key=lambda item: (item.objective, item.support))
    best = feasible[0]
    return ExhaustiveSparseResult(
        objective=best.objective,
        weights=best.weights.copy(),
        selectors=best.selectors,
        support=best.support,
        attempted_supports=attempted,
        feasible_supports=len(feasible),
        solutions=tuple(feasible),
    )


__all__ = [
    "EnumeratedSupportSolution",
    "ExhaustiveSparseResult",
    "enumerate_sparse_supports",
]
