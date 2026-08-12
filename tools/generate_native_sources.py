"""Generate the optional Cython C sources for the first-order solvers.

This is a maintainer tool.  End users build the checked-in generated C files
and therefore do not need Cython installed.  Each extension is generated from
the canonical Python implementation, so the Python and compiled solvers share
the same line search, restart, warm-start, and certificate logic.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Tuple


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ExtensionSource:
    module: str
    source: Path
    output: Path
    replacements: Tuple[Tuple[str, str], ...] = ()


SOURCES = (
    ExtensionSource(
        "sksfolio.relaxation.fista._native_budget_prox",
        ROOT / "sksfolio/relaxation/fista/budget_prox.py",
        ROOT / "sksfolio/relaxation/fista/_native_budget_prox.c",
    ),
    ExtensionSource(
        "sksfolio.relaxation.fista._native_linear_prox",
        ROOT / "sksfolio/relaxation/fista/linear_prox.py",
        ROOT / "sksfolio/relaxation/fista/_native_linear_prox.c",
        ((
            "from .budget_prox import prox_budget_details",
            "from ._native_budget_prox import prox_budget_details",
        ),),
    ),
    ExtensionSource(
        "sksfolio.relaxation.pdhg._native_safe_dual",
        ROOT / "sksfolio/relaxation/pdhg/safe_dual.py",
        ROOT / "sksfolio/relaxation/pdhg/_native_safe_dual.c",
        ((
            "from .solver import _prepare_problem",
            "from ._native_solver import _prepare_problem",
        ),),
    ),
    ExtensionSource(
        "sksfolio.relaxation.pdhg._native_solver",
        ROOT / "sksfolio/relaxation/pdhg/solver.py",
        ROOT / "sksfolio/relaxation/pdhg/_native_solver.c",
        ((
            "from .safe_dual import _dual_bound, evaluate_dual_bound",
            "from ._native_safe_dual import _dual_bound, evaluate_dual_bound",
        ),),
    ),
    ExtensionSource(
        "sksfolio.relaxation.fista._native_solver",
        ROOT / "sksfolio/relaxation/fista/solver.py",
        ROOT / "sksfolio/relaxation/fista/_native_solver.c",
        (
            (
                "from ..pdhg.safe_dual import SafeDualEvaluator",
                "from ..pdhg._native_safe_dual import SafeDualEvaluator",
            ),
            (
                "from .linear_prox import LinearConstraintProx",
                "from ._native_linear_prox import LinearConstraintProx",
            ),
        ),
    ),
    ExtensionSource(
        "sksfolio.relaxation.scsdg._native_solver",
        ROOT / "sksfolio/relaxation/scsdg/solver.py",
        ROOT / "sksfolio/relaxation/scsdg/_native_solver.c",
        (
            (
                "from ..pdhg.safe_dual import _dual_bound",
                "from ..pdhg._native_safe_dual import _dual_bound",
            ),
            (
                "from ..pdhg.solver import (",
                "from ..pdhg._native_solver import (",
            ),
        ),
    ),
)


def _generate(
    specification: ExtensionSource,
    staging_directory: Path,
) -> None:
    text = specification.source.read_text(encoding="utf-8")
    for original, replacement in specification.replacements:
        count = text.count(original)
        if count == 0:
            raise RuntimeError(
                f"replacement marker is absent from {specification.source}: "
                f"{original!r}"
            )
        text = text.replace(original, replacement)
    staging_directory.mkdir(parents=True, exist_ok=True)
    staged = staging_directory / (
        specification.module.replace(".", "_") + ".py"
    )
    staged.write_text(text, encoding="utf-8")
    specification.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "cython",
            "-3",
            "-X",
            "infer_types=True",
            "--module-name",
            specification.module,
            "-o",
            str(specification.output),
            str(staged.relative_to(ROOT)),
        ],
        cwd=ROOT,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="regenerate in a temporary tree and verify committed C files",
    )
    arguments = parser.parse_args()
    staging = ROOT / "build/cython-stage"
    check_root = ROOT / "build/cython-check"
    try:
        if not arguments.check:
            for specification in SOURCES:
                _generate(specification, staging)
            return
        for specification in SOURCES:
            generated = check_root / specification.output.relative_to(ROOT)
            check_specification = ExtensionSource(
                specification.module,
                specification.source,
                generated,
                specification.replacements,
            )
            _generate(check_specification, staging)
            if not specification.output.exists():
                raise SystemExit(f"missing {specification.output}")
            if generated.read_bytes() != specification.output.read_bytes():
                raise SystemExit(
                    f"generated source is stale: {specification.output}"
                )
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(check_root, ignore_errors=True)


if __name__ == "__main__":
    main()
