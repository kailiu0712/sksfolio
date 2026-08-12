"""Lazy bridge to the shared generic Julia/JuMP implementation."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import time
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from .problem import MarkowitzInstance, save_instance_bundle


JULIA_DIRECTORY = Path(__file__).resolve().parent / "_julia"
JULIA_SOURCE = JULIA_DIRECTORY / "MarkowitzCommercial.jl"
_PROJECT_INSTANTIATED = False
_PROJECT_LOCK = threading.Lock()


def _as_mapping(value: Any) -> Dict[str, Any]:
    try:
        items = value.items()
    except AttributeError as error:
        raise TypeError("Julia result must be a mapping") from error
    return {str(key): item for key, item in items}


def _load_module(instantiate: bool) -> Any:
    if instantiate:
        global _PROJECT_INSTANTIATED
        with _PROJECT_LOCK:
            if not _PROJECT_INSTANTIATED:
                try:
                    juliapkg = importlib.import_module("juliapkg")
                    executable = str(juliapkg.executable())
                except (ImportError, OSError) as error:
                    raise RuntimeError(
                        "cannot locate Julia to instantiate the solver "
                        "project"
                    ) from error
                subprocess.run(
                    [
                        executable,
                        f"--project={JULIA_DIRECTORY}",
                        "--startup-file=no",
                        "-e",
                        "import Pkg; Pkg.resolve(); Pkg.instantiate()",
                    ],
                    check=True,
                )
                _PROJECT_INSTANTIATED = True
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


def _solve_binary_bundle(
    bundle_path: Path,
    solver: str,
    options: Mapping[str, Any],
    required_assets: Sequence[int],
    forbidden_assets: Sequence[int],
    instantiate: bool,
) -> Dict[str, Any]:
    module = _load_module(instantiate)
    result = _as_mapping(
        module.solve_binary_bundle(
            str(bundle_path),
            solver,
            dict(options),
            [int(value) for value in required_assets],
            [int(value) for value in forbidden_assets],
        )
    )
    for key in ("x", "selectors"):
        if result.get(key) is not None:
            result[key] = np.asarray(
                result[key],
                dtype=float,
            ).reshape(-1)
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


def solve_binary_julia(
    problem: MarkowitzInstance,
    solver: str,
    options: Optional[Mapping[str, Any]] = None,
    *,
    required_assets: Sequence[int] = (),
    forbidden_assets: Sequence[int] = (),
) -> Dict[str, Any]:
    """Solve the binary perspective model through the shared JuMP module."""
    started = time.perf_counter()
    supplied = dict(options or {})
    instantiate = bool(supplied.pop("julia_instantiate", False))
    try:
        if problem.bundle_path is not None:
            return _solve_binary_bundle(
                Path(problem.bundle_path),
                solver,
                supplied,
                required_assets,
                forbidden_assets,
                instantiate,
            )
        with tempfile.TemporaryDirectory(
            prefix="sksfolio-julia-binary-"
        ) as temporary:
            bundle = save_instance_bundle(
                problem,
                Path(temporary) / "problem",
            )
            return _solve_binary_bundle(
                bundle,
                solver,
                supplied,
                required_assets,
                forbidden_assets,
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
            "formulation": "exact_binary_perspective_conic",
            "status": "unavailable" if unavailable else "error",
            "success": False,
            "has_solution": False,
            "x": None,
            "selectors": None,
            "message": message,
            "failure_kind": (
                "runtime_or_license" if unavailable else "solver"
            ),
            "total_seconds": time.perf_counter() - started,
        }


__all__ = [
    "JULIA_DIRECTORY",
    "JULIA_SOURCE",
    "solve_binary_julia",
    "solve_julia",
]
