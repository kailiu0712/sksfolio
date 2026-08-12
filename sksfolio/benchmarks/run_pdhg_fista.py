"""Run a matched PDHG--FISTA comparison on one synthetic BCW shape."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .run_relaxations import _main


MATCHED_BACKENDS = ("pdhg", "fista", "gurobi.python")
MATCHED_OUTPUT = Path("bcw_matched_pdhg_fista_results.csv")
MATCHED_TOLERANCE = 1e-8


def main(arguments: Sequence[str] | None = None) -> None:
    """Run both first-order methods and an optional optimal reference."""
    _main(
        arguments,
        default_backends=MATCHED_BACKENDS,
        default_output=MATCHED_OUTPUT,
        default_tolerance=MATCHED_TOLERANCE,
        budget_only=True,
    )


if __name__ == "__main__":
    main()
