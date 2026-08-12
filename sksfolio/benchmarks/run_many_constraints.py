"""Compare relaxation solvers on the full many-row portfolio model."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .run_relaxations import _main


MANY_CONSTRAINT_BACKENDS = ("pdhg", "fista", "gurobi.python")
MANY_CONSTRAINT_ORACLES = ("dual_fista", "majorization_qp")
MANY_CONSTRAINT_OUTPUT = Path("bcw_many_constraint_results.csv")


def main(arguments: Sequence[str] | None = None) -> None:
    """Run the 262-row constrained S&P-shaped benchmark by default."""
    _main(
        arguments,
        default_backends=MANY_CONSTRAINT_BACKENDS,
        default_output=MANY_CONSTRAINT_OUTPUT,
        default_tolerance=1e-5,
        budget_only=False,
        default_variants=("metric-linesearch-restart",),
        default_pava=("partial_sort",),
        default_fista_prox_oracles=MANY_CONSTRAINT_ORACLES,
        default_constraint_profile="many",
        default_regime="constrained",
    )


if __name__ == "__main__":
    main()
