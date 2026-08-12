"""Optional full-cardinality Gurobi reference and polishing backend."""

from __future__ import annotations

import math
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ..relaxation.problem import MarkowitzInstance
from .evaluation import evaluate_incumbent
from .result import IncumbentResult
from .restricted_qp import solve_restricted_qp
from .state import IncumbentState
from .support import validate_branch_indices


def _warm_values(value: Any, dimension: int) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if value is None:
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
        raise ValueError("Gurobi warm start has the wrong dimension")
    z = (
        (np.abs(x) > 1e-9).astype(float)
        if selectors is None
        else np.asarray(selectors, dtype=float).reshape(-1)
    )
    if z.shape != (dimension,):
        raise ValueError("Gurobi selector warm start has the wrong dimension")
    return x, z


def solve_gurobi_incumbent(
    instance: MarkowitzInstance,
    *,
    warm_start: Optional[Any] = None,
    required_assets: Sequence[int] = (),
    forbidden_assets: Sequence[int] = (),
    time_limit: Optional[float] = None,
    options: Optional[Mapping[str, Any]] = None,
) -> IncumbentResult:
    """Solve or time-limit the original binary sparse QP with Gurobi.

    This is a reference backend, not part of the solver-free heuristic
    default. The default formulation adds the perspective inequalities
    ``x[i] ** 2 <= t[i] * z[i]``. Set ``formulation='miqp'`` in ``options``
    to request the weaker activation-only MIQP. Any returned portfolio is
    independently rechecked before its objective is exposed as an upper
    bound.
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as error:
        raise ImportError(
            "gurobipy is required for solve_gurobi_incumbent"
        ) from error

    instance.validate()
    required, forbidden = validate_branch_indices(
        instance.dimension,
        instance.k,
        required_assets,
        forbidden_assets,
    )
    settings = dict(options or {})
    polish_incumbent = bool(settings.pop("polish_incumbent", True))
    polish_options = dict(settings.pop("polish_options", {}))
    formulation = str(
        settings.pop("formulation", "perspective")
    ).lower().replace("-", "_")
    if formulation not in {"perspective", "miqp"}:
        raise ValueError("formulation must be perspective or miqp")
    start = time.perf_counter()
    try:
        model = gp.Model("sksfolio_sparse_incumbent")
    except Exception as error:
        return IncumbentResult(
            {
                "status": "unavailable",
                "solver": "gurobi",
                "x": None,
                "selectors": None,
                "upper_bound": None,
                "numerically_feasible": False,
                "error": f"{type(error).__name__}: {error}",
                "solve_seconds": time.perf_counter() - start,
                "state": None,
            }
        )
    model.Params.OutputFlag = int(bool(settings.pop("verbose", False)))
    if time_limit is not None:
        model.Params.TimeLimit = max(float(time_limit), 0.0)
    defaults = {
        "MIPFocus": 1,
        "MIPGap": 1e-4,
    }
    for name, value in defaults.items():
        if name not in settings:
            setattr(model.Params, name, value)
    for name, value in settings.items():
        setattr(model.Params, str(name), value)

    dimension = instance.dimension
    rank = instance.rank
    x = model.addMVar(dimension, lb=0.0, ub=1.0, name="x")
    z = model.addMVar(dimension, vtype=GRB.BINARY, name="z")
    exposure = model.addMVar(rank, lb=-GRB.INFINITY, name="factor")
    perspective_epigraph = (
        model.addMVar(dimension, lb=0.0, name="perspective_epigraph")
        if formulation == "perspective"
        else None
    )
    model.addConstr(x <= z, name="activation")
    model.addConstr(z.sum() <= int(instance.k), name="cardinality")
    if required.size:
        model.addConstr(z[required] == 1.0, name="required")
    if forbidden.size:
        model.addConstr(z[forbidden] == 0.0, name="forbidden")
    model.addConstr(
        exposure == np.asarray(instance.B, dtype=float).T @ x,
        name="factor_definition",
    )
    if perspective_epigraph is not None:
        model.addConstr(
            x * x <= perspective_epigraph * z,
            name="perspective",
        )
    if instance.rows:
        row_values = instance.C @ x
        lower = np.asarray(instance.lower, dtype=float)
        upper = np.asarray(instance.upper, dtype=float)
        finite_lower = np.flatnonzero(np.isfinite(lower))
        finite_upper = np.flatnonzero(np.isfinite(upper))
        if finite_lower.size:
            model.addConstr(
                row_values[finite_lower] >= lower[finite_lower],
                name="row_lower",
            )
        if finite_upper.size:
            model.addConstr(
                row_values[finite_upper] <= upper[finite_upper],
                name="row_upper",
            )
    ridge_term = (
        perspective_epigraph.sum()
        if perspective_epigraph is not None
        else x @ x
    )
    objective = (
        0.5 * (exposure @ exposure)
        + 0.5 * float(instance.perspective_weight) * ridge_term
        - float(instance.return_reward)
        * (np.asarray(instance.mu, dtype=float) @ x)
    )
    model.setObjective(objective, GRB.MINIMIZE)
    warm_x, warm_z = _warm_values(warm_start, dimension)
    if warm_x is not None and warm_z is not None:
        clipped_x = np.clip(warm_x, 0.0, 1.0)
        rounded_z = np.clip(np.rint(warm_z), 0.0, 1.0)
        x.Start = clipped_x
        z.Start = rounded_z
        if perspective_epigraph is not None:
            perspective_epigraph.Start = np.divide(
                clipped_x * clipped_x,
                rounded_z,
                out=np.zeros_like(clipped_x),
                where=rounded_z > 0.5,
            )
    build_seconds = time.perf_counter() - start
    optimize_start = time.perf_counter()
    model.optimize()
    optimizer_seconds = time.perf_counter() - optimize_start
    elapsed = time.perf_counter() - start
    status_map = {
        GRB.OPTIMAL: "optimal",
        GRB.TIME_LIMIT: "time_limit",
        GRB.INTERRUPTED: "interrupted",
        GRB.INFEASIBLE: "infeasible",
        GRB.INF_OR_UNBD: "infeasible_or_unbounded",
    }
    status = status_map.get(model.Status, f"gurobi_status_{model.Status}")
    objective_bound = (
        float(model.ObjBound)
        if model.SolCount > 0 or math.isfinite(float(model.ObjBound))
        else None
    )
    if model.SolCount <= 0:
        return IncumbentResult(
            {
                "status": status,
                "solver": "gurobi",
                "formulation": (
                    "exact_binary_perspective_miqcp"
                    if formulation == "perspective"
                    else "exact_binary_miqp"
                ),
                "x": None,
                "selectors": None,
                "upper_bound": None,
                "numerically_feasible": False,
                "solver_objective_bound": objective_bound,
                "build_seconds": build_seconds,
                "optimizer_seconds": optimizer_seconds,
                "solve_seconds": elapsed,
                "total_seconds": elapsed,
                "state": None,
            }
        )
    weights = np.asarray(x.X, dtype=float)
    selectors = np.clip(np.rint(np.asarray(z.X, dtype=float)), 0.0, 1.0)
    raw_incumbent_objective = float(model.ObjVal)
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
    tolerance = float(settings.get("FeasibilityTol", 1e-6))
    diagnostics = evaluate_incumbent(
        instance,
        weights,
        selectors,
        feasibility_tolerance=max(tolerance, 1e-8),
        required_assets=required,
        forbidden_assets=forbidden,
    )
    upper_bound = diagnostics["upper_bound"]
    state = (
        IncumbentState(
            dimension=dimension,
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
    return IncumbentResult(
        {
            "status": status,
            "solver": "gurobi",
            "formulation": (
                "exact_binary_perspective_miqcp"
                if formulation == "perspective"
                else "exact_binary_miqp"
            ),
            "x": weights,
            "selectors": selectors,
            "upper_bound": upper_bound,
            "numerically_feasible": bool(
                diagnostics["numerically_feasible"]
            ),
            "floating_point_certified": False,
            "solver_objective_bound": objective_bound,
            "solver_mip_gap": float(model.MIPGap),
            "node_count": float(model.NodeCount),
            "solution_count": int(model.SolCount),
            "solver_incumbent_objective": raw_incumbent_objective,
            "incumbent_polish_status": polish_status,
            "incumbent_polish_seconds": polish_seconds,
            "diagnostics": diagnostics,
            "build_seconds": build_seconds,
            "optimizer_seconds": optimizer_seconds,
            "solve_seconds": elapsed,
            "total_seconds": time.perf_counter() - start,
            "state": state,
        }
    )


__all__ = ["solve_gurobi_incumbent"]
