"""One entry point for all 17 custom variants and two MOSEK settings."""
from importlib import import_module
from time import perf_counter

CUSTOM_METHODS = (
    "pdhg", "adaptive_pdhg", "alm_apg",
    "newton_dense", "newton_krylov", "newton_krylov_inexact",
    "alm_original", "alm_fast", "alm_fast_tuned", "active_set", "alm_polished",
    "alm_preconditioned",
    "nested_fast_prox", "nested_fast_outer", "nested_fast", "nested_fast_inexact",
    "ipm",
)
MOSEK_METHODS = ("mosek_fast", "mosek_reference")


def solve(problem, method="alm_polished", time_limit=15.0, tol=1e-6,
          feas_tol=1e-7, max_iter=None):
    """Return holdings, multipliers, status, and independently checked metrics.

    Call warmup() before timing to exclude Numba compilation.
    Time limits are cooperative and checked between numerical operations.
    """
    if method not in CUSTOM_METHODS + MOSEK_METHODS:
        raise ValueError(f"Unknown method {method!r}")
    if time_limit <= 0 or tol <= 0 or feas_tol <= 0:
        raise ValueError("time_limit, tol, and feas_tol must be positive")
    if method in MOSEK_METHODS:
        from .reference import solve_mosek
        internal = 1e-7 if method == "mosek_fast" else 1e-9
        result = solve_mosek(problem, time_limit=time_limit, tol=internal)
    else:
        if method in CUSTOM_METHODS[:3]:
            module, mode, cap = "first_order", method, 1000000
        elif method.startswith("newton_"):
            module, mode, cap = "newton", method.removeprefix("newton_"), 1000
        elif method == "alm_original":
            module, mode, cap = "alm_newton", "alm_newton", 5000
        elif method.startswith("alm_fast"):
            module, mode, cap = "improvements.alm_fast", method, 5000
        elif method.startswith("nested_fast"):
            module, mode, cap = "improvements.nested_fast", method, 5000
        else:
            module, mode, cap = "improvements." + method, method, 5000
        solver = import_module("." + module, package="sparsefolio").solve
        result = solver(problem, mode=mode, time_limit=time_limit, tol=tol,
                        feas_tol=feas_tol, max_iter=cap if max_iter is None else max_iter)
    from .core import evaluate
    result["metrics"] = evaluate(problem, result["x"], result["y"])
    result["solved"] = bool(result["metrics"]["rel_gap"] <= tol
                            and result["metrics"]["feasibility"] <= feas_tol)
    return result


def warmup():
    """Compile or load every timed Numba kernel and report nopython signatures."""
    import numpy as np
    import numba
    from . import core
    from .improvements.fast_core import prox as fast_prox
    if numba.config.DISABLE_JIT:
        raise RuntimeError("Numba JIT is disabled. Unset NUMBA_DISABLE_JIT.")
    v = np.array([0.5, 0.8, -0.2], dtype=np.float64)
    gamma, upper = np.ones(3), np.ones(3)
    start = perf_counter()
    original_start = perf_counter()
    core.prox(v, 0.5, gamma, upper, 1)
    original_seconds = perf_counter() - original_start
    fast_start = perf_counter()
    fast_prox(v, 0.5, gamma, upper, 1)
    fast_seconds = perf_counter() - fast_start
    value_start = perf_counter()
    core.value_and_z(np.array([0.1, 0.2, 0.3]), gamma, upper, 1)
    value_seconds = perf_counter() - value_start
    kernels = {"coordinate_prox": core._at_theta, "original_prox": core.prox,
               "fast_prox": fast_prox, "perspective_value": core.value_and_z}
    signatures = {name: [str(s) for s in fn.nopython_signatures]
                  for name, fn in kernels.items()}
    if any(not signatures[name] for name in ("original_prox", "fast_prox", "perspective_value")):
        raise RuntimeError("Expected Numba nopython compilation was not observed")
    return dict(seconds=perf_counter()-start, original_prox_seconds=original_seconds,
                fast_prox_seconds=fast_seconds, value_seconds=value_seconds,
                jit_enabled=True, nopython_signatures=signatures)
