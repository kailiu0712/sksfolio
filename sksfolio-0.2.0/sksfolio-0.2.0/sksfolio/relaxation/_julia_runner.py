"""Lazy bridge to the shared Julia/JuMP commercial implementation."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any, Dict, Mapping, Optional

import numpy as np

from .problem import MarkowitzInstance, save_instance_bundle


JULIA_DIRECTORY = Path(__file__).resolve().parent / "_julia"
JULIA_SOURCE = JULIA_DIRECTORY / "MarkowitzCommercial.jl"


def _as_mapping(value: Any) -> Dict[str, Any]:
    try:
        items = value.items()
    except AttributeError as error:
        raise TypeError("Julia result must be a mapping") from error
    return {str(key): item for key, item in items}


def _load_module(instantiate: bool) -> Any:
    if instantiate:
        try:
            juliapkg = importlib.import_module("juliapkg")
            executable = str(juliapkg.executable())
        except (ImportError, OSError) as error:
            raise RuntimeError(
                "cannot locate Julia to instantiate the solver project"
            ) from error
        subprocess.run(
            [
                executable,
                f"--project={JULIA_DIRECTORY}",
                "--startup-file=no",
                "-e",
                "import Pkg; Pkg.instantiate()",
            ],
            check=True,
        )
    try:
        juliacall = importlib.import_module("juliacall")
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "the Julia backend requires juliacall and Julia"
        ) from error
    julia = juliacall.Main
    project_literal = json.dumps(str(JULIA_DIRECTORY))
    julia.seval(
        "if !("
        + project_literal
        + " in LOAD_PATH); pushfirst!(LOAD_PATH, "
        + project_literal
        + "); end"
    )
    module_loaded = bool(
        julia.seval("isdefined(Main, :MarkowitzCommercial)")
    )
    if not module_loaded:
        julia.include(str(JULIA_SOURCE))
    return julia.MarkowitzCommercial


def _solve_bundle(
    bundle_path: Path,
    solver: str,
    options: Mapping[str, Any],
    instantiate: bool,
) -> Dict[str, Any]:
    module = _load_module(instantiate)
    result = _as_mapping(
        module.solve_bundle(
            str(bundle_path),
            solver,
            dict(options),
        )
    )
    if result.get("x") is not None:
        result["x"] = np.asarray(result["x"], dtype=float).reshape(-1)
    return result


def solve_julia(
    problem: MarkowitzInstance,
    solver: str,
    options: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Solve a problem through JuMP without copying large arrays to Julia."""
    started = time.perf_counter()
    supplied = dict(options or {})
    instantiate = bool(supplied.pop("julia_instantiate", False))
    try:
        if problem.bundle_path is not None:
            return _solve_bundle(
                Path(problem.bundle_path),
                solver,
                supplied,
                instantiate,
            )
        with tempfile.TemporaryDirectory(
            prefix="sksfolio-julia-"
        ) as temporary:
            bundle = save_instance_bundle(
                problem,
                Path(temporary) / "problem",
            )
            return _solve_bundle(
                bundle,
                solver,
                supplied,
                instantiate,
            )
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        lowered = message.lower()
        unavailable = any(
            marker in lowered
            for marker in (
                "juliacall is unavailable",
                "requires juliacall",
                "cannot locate julia",
                "could not find julia",
                "julia executable",
                "juliapkg is unavailable",
                "package not found",
                "package is required",
                "could not load library",
                "license",
                "permission",
            )
        )
        return {
            "solver": solver,
            "language": "julia",
            "status": "unavailable" if unavailable else "error",
            "success": False,
            "has_solution": False,
            "x": None,
            "message": message,
            "failure_kind": (
                "runtime_or_license" if unavailable else "solver"
            ),
            "total_seconds": time.perf_counter() - started,
        }


__all__ = ["JULIA_DIRECTORY", "JULIA_SOURCE", "solve_julia"]
