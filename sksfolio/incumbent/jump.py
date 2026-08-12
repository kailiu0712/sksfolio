"""Configurable Julia/JuMP binary perspective reference backend."""

from __future__ import annotations

import math
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ..relaxation._julia_runner import solve_binary_julia
from ..relaxation.jump.julia import _optimizer_specification
from ..relaxation.problem import MarkowitzInstance
from .evaluation import evaluate_incumbent
from .restricted_qp import solve_restricted_qp
from .result import IncumbentResult
from .state import IncumbentState
from .support import validate_branch_indices


def _warm_values(
    value: Any,
    dimension: int,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if value is None or value is False:
        return None, None
    weights = getattr(value, "weights", None)
    selectors = getattr(value, "selectors", None)
    raw = getattr(value, "raw", value)
    if isinstance(raw, Mapping):
        if weights is None:
            weights = raw.get("x")
        if selectors is None:
            selectors = raw.get("selectors")
    if weights is None:
        return None, None
    x = np.asarray(weights, dtype=float).reshape(-1)
    if x.shape != (dimension,):
        raise ValueError("JuMP warm start has the wrong dimension")
    z = (
        (np.abs(x) > 1e-9).astype(float)
        if selectors is None
        else np.asarray(selectors, dtype=float).reshape(-1)
    )
    if z.shape != (dimension,):
        raise ValueError("JuMP selector warm start has the wrong dimension")
    return x, z


def solve_jump_incumbent(
    instance: MarkowitzInstance,
    *,
    optimizer: str = "gurobi",
    warm_start: Optional[Any] = None,
    required_assets: Sequence[int] = (),
    forbidden_assets: Sequence[int] = (),
    time_limit: Optional[float] = None,
    options: Optional[Mapping[str, Any]] = None,
) -> IncumbentResult:
    """Solve the exact binary perspective model through Julia/JuMP.

    ``optimizer`` accepts the same aliases and ``Package.Constructor``
    syntax as the continuous JuMP backend. In practice, the model requires
    a mixed-integer rotated-cone optimizer such as Gurobi or MOSEK.
    """
    started = time.perf_counter()
    instance.validate()
    required, forbidden = validate_branch_indices(
        instance.dimension,
        instance.k,
        required_assets,
        forbidden_assets,
    )
    optimizer_name = _optimizer_specification({"optimizer": optimizer})
    settings = dict(options or {})
    polish_incumbent = bool(settings.pop("polish_incumbent", True))
    polish_options = dict(settings.pop("polish_options", {}))
    if time_limit is not None:
        if not math.isfinite(float(time_limit)) or float(time_limit) <= 0.0:
            raise ValueError("time_limit must be positive and finite")
        if "time_limit" in settings:
            raise ValueError(
                "pass time_limit as a named argument or in options, not both"
            )
        settings["time_limit"] = float(time_limit)
    settings.setdefault("tolerance", 1e-8)
    warm_x, warm_z = _warm_values(warm_start, instance.dimension)
    if warm_x is not None:
        if "initial_x" in settings or "initial_z" in settings:
            raise ValueError(
                "pass warm_start or initial_x/initial_z, not both"
            )
        settings["warm_start"] = True
        settings["initial_x"] = warm_x.tolist()
        settings["initial_z"] = warm_z.tolist()
    else:
        settings.setdefault("warm_start", False)

    raw = dict(
        solve_binary_julia(
            instance,
            optimizer_name,
            settings,
            required_assets=required.tolist(),
            forbidden_assets=forbidden.tolist(),
        )
    )
    raw.setdefault("solver", optimizer_name)
    raw.setdefault("language", "julia")
    raw.setdefault("formulation", "exact_binary_perspective_conic")
    raw.setdefault("total_seconds", time.perf_counter() - started)
    details = raw.get("solver_details", {})
    if raw.get("solver_objective_bound") is None and isinstance(
        details,
        Mapping,
    ):
        raw["solver_objective_bound"] = details.get("objective_bound")
    weights_value = raw.get("x")
    selectors_value = raw.get("selectors")
    if weights_value is None or selectors_value is None:
        raw.update(
            {
                "x": None,
                "selectors": None,
                "upper_bound": None,
                "numerically_feasible": False,
                "state": None,
            }
        )
        return IncumbentResult(raw)

    weights = np.asarray(weights_value, dtype=float).reshape(-1)
    selectors = np.clip(
        np.rint(np.asarray(selectors_value, dtype=float).reshape(-1)),
        0.0,
        1.0,
    )
    polish_status = "disabled"
    polish_seconds = 0.0
    if polish_incumbent:
        polish_start = time.perf_counter()
        polished = solve_restricted_qp(
            instance,
            np.flatnonzero(selectors > 0.5),
            solver="osqp",
            warm_start=weights,
            options={
                "eps_abs": 1e-10,
                "eps_rel": 1e-10,
                "feasibility_tolerance": 1e-7,
                **polish_options,
            },
        )
        polish_seconds = time.perf_counter() - polish_start
        polish_status = polished.status
        if polished.feasible:
            weights = polished.x
    diagnostics = evaluate_incumbent(
        instance,
        weights,
        selectors,
        feasibility_tolerance=1e-7,
        required_assets=required,
        forbidden_assets=forbidden,
    )
    upper_bound = diagnostics["upper_bound"]
    state = (
        IncumbentState(
            dimension=instance.dimension,
            k=instance.k,
            constraint_ids=tuple(instance.constraint_names),
            x=weights,
            selectors=selectors,
            objective=upper_bound,
            required_assets=tuple(int(value) for value in required),
            forbidden_assets=tuple(int(value) for value in forbidden),
        ).to_dict(copy=False)
        if diagnostics["numerically_feasible"]
        else None
    )
    raw.update(
        {
            "x": weights,
            "selectors": selectors,
            "upper_bound": upper_bound,
            "numerically_feasible": bool(
                diagnostics["numerically_feasible"]
            ),
            "floating_point_certified": False,
            "incumbent_polish_status": polish_status,
            "incumbent_polish_seconds": polish_seconds,
            "diagnostics": diagnostics,
            "state": state,
            "total_seconds": time.perf_counter() - started,
        }
    )
    return IncumbentResult(raw)


__all__ = ["solve_jump_incumbent"]
