"""Discovery and loading of the optional compiled first-order solvers."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Dict, Tuple


FIRST_ORDER_BACKENDS = ("fista", "pdhg", "scsdg")
IMPLEMENTATIONS = ("auto", "native", "python")
# Full-module Cython compilation showed no robust warm-runtime advantage in
# matched d=1000 tests because NumPy/BLAS, SciPy L-BFGS-B, sparse products,
# and partial-sort PAVA are already native.  Keep this explicit and easy to
# revise after a truly typed/GIL-free core is benchmarked.
AUTO_IMPLEMENTATION = {
    backend: "python" for backend in FIRST_ORDER_BACKENDS
}
_MODULES = {
    backend: f"sksfolio.relaxation.{backend}._native_solver"
    for backend in FIRST_ORDER_BACKENDS
}


def _normalize_implementation(value: str) -> str:
    implementation = str(value).strip().lower().replace("-", "_")
    implementation = {
        "c": "native",
        "compiled": "native",
        "cython": "native",
        "reference": "python",
    }.get(implementation, implementation)
    if implementation not in IMPLEMENTATIONS:
        raise ValueError(
            "implementation must be 'auto', 'native', or 'python'"
        )
    return implementation


def native_available(backend: str) -> bool:
    """Return whether the complete compiled backend can be imported."""
    name = str(backend).lower()
    if name not in _MODULES:
        return False
    try:
        importlib.import_module(_MODULES[name])
    except (ImportError, OSError):
        return False
    return True


def native_availability() -> Dict[str, bool]:
    """Return availability for all three compiled solver families."""
    return {
        backend: native_available(backend)
        for backend in FIRST_ORDER_BACKENDS
    }


def load_first_order_module(
    backend: str,
    implementation: str = "auto",
) -> Tuple[ModuleType, str]:
    """Load a solver module and report its actual implementation.

    ``auto`` uses the matched-benchmark policy in :data:`AUTO_IMPLEMENTATION`.
    A forced ``native`` request never falls back silently.
    """
    family = str(backend).lower()
    if family not in FIRST_ORDER_BACKENDS:
        raise ValueError(
            "compiled implementations exist only for fista, pdhg, and scsdg"
        )
    requested = _normalize_implementation(implementation)
    if requested == "auto":
        requested = AUTO_IMPLEMENTATION[family]
    if requested == "native":
        try:
            return importlib.import_module(_MODULES[family]), "native"
        except (ImportError, OSError) as error:
            raise RuntimeError(
                f"the compiled {family} extension is unavailable; "
                "install a platform wheel or build the package with a "
                "C compiler"
            ) from error
    return (
        importlib.import_module(
            f"sksfolio.relaxation.{family}.solver"
        ),
        "python",
    )


__all__ = [
    "FIRST_ORDER_BACKENDS",
    "AUTO_IMPLEMENTATION",
    "IMPLEMENTATIONS",
    "load_first_order_module",
    "native_availability",
    "native_available",
]
