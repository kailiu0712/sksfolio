"""Compare the four proximal strategies and native Gurobi."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .run_relaxations import _main


FOUR_PROX_BACKENDS = ("pdhg", "fista", "gurobi.python")
FOUR_PROX_ORACLES = (
    "budget",
    "dual_fista",
    "majorization_qp",
)
FOUR_PROX_OUTPUT = Path("bcw_four_prox_results.csv")


def main(arguments: Sequence[str] | None = None) -> None:
    """Run PAVA, budget Brent/PAVA, dual FISTA, and majorization QP."""
    _main(
        arguments,
        default_backends=FOUR_PROX_BACKENDS,
        default_output=FOUR_PROX_OUTPUT,
        default_tolerance=1e-6,
        budget_only=False,
        default_variants=("metric-linesearch-restart",),
        default_pava=("partial_sort",),
        default_fista_prox_oracles=FOUR_PROX_ORACLES,
    )


if __name__ == "__main__":
    main()
