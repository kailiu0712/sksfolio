from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = WORKSPACE_ROOT / "sksfolio-0.2.0" / "sksfolio-0.2.0"
VSWHERE = Path(
    r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
)
VENV_PYTHON = WORKSPACE_ROOT / ".venv" / "Scripts" / "python.exe"


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    detail: str
    fix: str | None = None


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _add(
    results: list[CheckResult],
    name: str,
    ok: bool,
    detail: str,
    fix: str | None = None,
) -> None:
    results.append(CheckResult(name, "OK" if ok else "MISSING", detail, fix))


def _format(results: list[CheckResult]) -> str:
    name_width = max(len(item.name) for item in results)
    status_width = max(len(item.status) for item in results)
    lines = []
    for item in results:
        line = (
            f"{item.name.ljust(name_width)}  "
            f"{item.status.ljust(status_width)}  "
            f"{item.detail}"
        )
        lines.append(line)
        if item.fix and item.status != "OK":
            lines.append(" " * (name_width + status_width + 4) + f"fix: {item.fix}")
    return "\n".join(lines)


def main() -> int:
    results: list[CheckResult] = []

    _add(
        results,
        "workspace",
        PROJECT_ROOT.is_dir(),
        f"project root {'found' if PROJECT_ROOT.is_dir() else 'missing'}: {PROJECT_ROOT}",
        "run this script from the repository workspace that contains sksfolio-0.2.0",
    )

    _add(
        results,
        "python",
        sys.version_info >= (3, 10),
        f"{sys.executable} [{sys.version.split()[0]}]",
        "use Python 3.10+",
    )

    _add(
        results,
        "venv",
        sys.prefix != sys.base_prefix,
        f"prefix={sys.prefix}",
        r"use .\.venv\Scripts\Activate.ps1 before installing or benchmarking",
    )

    core_modules = ("numpy", "scipy", "threadpoolctl", "sksfolio")
    for module_name in core_modules:
        available = _module_available(module_name)
        fix = None
        if not available:
            if module_name == "sksfolio":
                fix = rf'& "{VENV_PYTHON}" -m pip install -e "{PROJECT_ROOT}"'
            else:
                fix = rf'& "{VENV_PYTHON}" -m pip install -e "{PROJECT_ROOT}"'
        _add(
            results,
            module_name,
            available,
            "importable" if available else "not importable",
            fix,
        )

    try:
        version = importlib.metadata.version("sksfolio")
    except importlib.metadata.PackageNotFoundError:
        version = None
    _add(
        results,
        "editable-install",
        version is not None,
        f"sksfolio {version}" if version else "distribution metadata not found",
        rf'& "{VENV_PYTHON}" -m pip install -e "{PROJECT_ROOT}"',
    )

    native_available = False
    native_enabled = False
    native_error: str | None = None
    if _module_available("sksfolio"):
        try:
            pava = importlib.import_module("sksfolio.relaxation.pdhg.pava")
            native_available = bool(pava.native_partial_sort_available())
            native_enabled = bool(pava.native_partial_sort_enabled())
        except Exception as error:  # pragma: no cover - diagnostic path
            native_error = f"{type(error).__name__}: {error}"

    _add(
        results,
        "native-pava",
        native_available,
        (
            f"available={native_available}, enabled={native_enabled}"
            if native_error is None
            else native_error
        ),
        (
            "install Visual Studio Build Tools with Desktop development "
            "with C++, open a Developer PowerShell, then run "
            r'".\rebuild_native_pava.ps1"'
        ),
    )

    cl_path = shutil.which("cl")
    _add(
        results,
        "msvc-cl",
        cl_path is not None or native_available,
        (
            cl_path
            if cl_path
            else "cl.exe not in PATH for this shell; only needed for future rebuilds"
        ),
        (
            "open a Visual Studio Developer PowerShell, or install "
            "Build Tools 2022 with the C++ workload"
        ),
    )

    _add(
        results,
        "vswhere",
        VSWHERE.exists(),
        str(VSWHERE) if VSWHERE.exists() else "vswhere.exe not found",
        "install Visual Studio Build Tools 2022",
    )

    for module_name, extra_name in (
        ("osqp", "qp"),
        ("gurobipy", "gurobi"),
        ("mosek", "mosek"),
        ("juliacall", "julia"),
    ):
        available = _module_available(module_name)
        if extra_name == "qp":
            fix = rf'& "{VENV_PYTHON}" -m pip install -e "{PROJECT_ROOT}[qp]"'
        elif extra_name == "julia":
            fix = rf'& "{VENV_PYTHON}" -m pip install -e "{PROJECT_ROOT}[julia]"'
        else:
            fix = rf'& "{VENV_PYTHON}" -m pip install -e "{PROJECT_ROOT}[{extra_name}]"'
        _add(
            results,
            module_name,
            available,
            "importable" if available else "not importable",
            fix,
        )

    for env_name in ("GUROBI_HOME", "GRB_LICENSE_FILE", "MOSEKLM_LICENSE_FILE"):
        value = os.environ.get(env_name, "")
        _add(
            results,
            env_name,
            bool(value),
            value if value else "not set",
            f"set {env_name} before using the corresponding commercial solver",
        )

    console_scripts = {
        entry.name
        for entry in importlib.metadata.entry_points(group="console_scripts")
    }
    for command in (
        "sksfolio-bcw",
        "sksfolio-four-prox",
        "sksfolio-many-constraints",
        "sksfolio-prox-native",
        "sksfolio-fista-restarts",
    ):
        _add(
            results,
            command,
            command in console_scripts,
            "registered" if command in console_scripts else "not registered",
            rf'& "{VENV_PYTHON}" -m pip install -e "{PROJECT_ROOT}"',
        )

    print("sksfolio local environment doctor")
    print(f"workspace: {WORKSPACE_ROOT}")
    print()
    print(_format(results))

    missing = [item for item in results if item.status != "OK"]
    if missing:
        print()
        print("Summary")
        print("- Core Python fallback path is usable.")
        if not native_available:
            print("- Native partial-sort PAVA still needs a compiler-backed rebuild.")
        if not _module_available("osqp"):
            print("- The exact majorization-QP path still needs OSQP.")
        if not _module_available("gurobipy") or not _module_available("mosek"):
            print("- Commercial backends still need their Python packages.")
        if not os.environ.get("GRB_LICENSE_FILE") and not os.environ.get("GUROBI_HOME"):
            print("- Gurobi will also need a valid local license configuration.")
        if not os.environ.get("MOSEKLM_LICENSE_FILE"):
            print("- MOSEK will also need a valid local license configuration.")
        return 1

    print()
    print("All checked capabilities are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
