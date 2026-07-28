"""Native Gurobi implementation of the perspective relaxation.

The optional `gurobipy` package is imported only when :func:`solve` is
called. Missing packages and licenses are returned as normalized unavailable
results.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional

import numpy as np

from .._commercial_common import (
    _complete_solution_result,
    _empty_result,
    _failure_kind,
    _prepare_instance,
    _prepare_options,
    _warm_start_values,
)


def _gurobi_status(model: Any, GRB: Any) -> str:
    status_map = {
        GRB.OPTIMAL: "optimal",
        GRB.SUBOPTIMAL: "suboptimal",
        GRB.TIME_LIMIT: "time_limit",
        GRB.ITERATION_LIMIT: "iteration_limit",
        GRB.NODE_LIMIT: "node_limit",
        GRB.SOLUTION_LIMIT: "solution_limit",
        GRB.INTERRUPTED: "interrupted",
        GRB.NUMERIC: "numerical_error",
        GRB.INFEASIBLE: "infeasible",
        GRB.UNBOUNDED: "unbounded",
        GRB.INF_OR_UNBD: "infeasible_or_unbounded",
    }
    for name, label in (
        ("CUTOFF", "cutoff"),
        ("USER_OBJ_LIMIT", "objective_limit"),
        ("WORK_LIMIT", "work_limit"),
        ("MEM_LIMIT", "memory_limit"),
    ):
        code = getattr(GRB, name, None)
        if code is not None:
            status_map[code] = label
    return status_map.get(int(model.Status), f"status_{int(model.Status)}")


def _finite_model_attribute(model: Any, name: str) -> Optional[float]:
    try:
        value = float(getattr(model, name))
    except Exception:
        return None
    return value if math.isfinite(value) else None


def solve(
    instance: Any,
    options: Optional[Any] = None,
) -> Dict[str, Any]:
    """Solve an instance with native ``gurobipy``.

    Supported options are ``threads``, ``tolerance``, ``time_limit``, ``log``,
    ``warm_start``, and ``gurobi_params``.  The last item is a mapping of
    additional Gurobi parameter names to values.
    """
    total_start = time.perf_counter()
    try:
        prepared = _prepare_instance(instance)
        settings = _prepare_options(options, "gurobi_params")
    except Exception as error:
        return _empty_result(
            "gurobi",
            "error",
            total_start,
            f"{type(error).__name__}: {error}",
        )

    try:
        import gurobipy as gp
        from gurobipy import GRB
    except (ImportError, OSError) as error:
        return _empty_result(
            "gurobi",
            "unavailable",
            total_start,
            f"gurobipy is unavailable: {type(error).__name__}: {error}",
        )

    model: Any = None
    environment: Any = None
    try:
        environment = gp.Env(empty=True)
        environment.setParam("OutputFlag", int(settings["log"]))
        environment.start()
        model = gp.Model("markowitz_perspective", env=environment)
        model.Params.OutputFlag = int(settings["log"])
        model.Params.NonConvex = 0
        model.Params.Method = 2
        model.Params.Crossover = 0
        model.Params.Threads = settings["threads"]
        tolerance = settings["tolerance"]
        feasibility_tolerance = max(1e-9, min(1e-2, tolerance))
        model.Params.FeasibilityTol = feasibility_tolerance
        model.Params.OptimalityTol = feasibility_tolerance
        model.Params.BarConvTol = tolerance
        model.Params.BarQCPConvTol = tolerance
        if settings["time_limit"] is not None:
            model.Params.TimeLimit = settings["time_limit"]
        for name, value in settings["solver_params"].items():
            model.setParam(str(name), value)

        x = model.addMVar(
            prepared.dimension,
            lb=0.0,
            ub=1.0,
            name="x",
        )
        z = model.addMVar(
            prepared.dimension,
            lb=0.0,
            ub=1.0,
            name="z",
        )
        t = model.addMVar(
            prepared.dimension,
            lb=0.0,
            name="t",
        )
        exposure = model.addMVar(
            prepared.factors,
            lb=-GRB.INFINITY,
            name="factor_exposure",
        )

        model.addConstr(x <= z, name="long_only_link")
        model.addConstr(z.sum() <= prepared.k, name="perspective_budget")
        model.addConstr(x * x <= t * z, name="perspective_cones")
        model.addConstr(
            exposure == prepared.B.T @ x,
            name="factor_definition",
        )
        if prepared.constraints:
            cx = prepared.C @ x
            finite_lower = np.isfinite(prepared.lower)
            finite_upper = np.isfinite(prepared.upper)
            if np.any(finite_lower):
                model.addConstr(
                    cx[finite_lower] >= prepared.lower[finite_lower],
                    name="portfolio_lower",
                )
            if np.any(finite_upper):
                model.addConstr(
                    cx[finite_upper] <= prepared.upper[finite_upper],
                    name="portfolio_upper",
                )

        objective = (
            0.5 * (exposure @ exposure)
            + 0.5 * prepared.perspective_weight * t.sum()
            - prepared.return_reward * (prepared.mu @ x)
        )
        model.setObjective(objective, GRB.MINIMIZE)

        warm_start_used = False
        if settings["warm_start"]:
            start_values = _warm_start_values(prepared)
            if start_values is not None:
                x_start, z_start, t_start, exposure_start, _ = start_values
                x.Start = x_start
                z.Start = z_start
                t.Start = t_start
                exposure.Start = exposure_start
                warm_start_used = True

        model.update()
        build_seconds = time.perf_counter() - total_start
        solve_start = time.perf_counter()
        model.optimize()
        solve_seconds = time.perf_counter() - solve_start

        status = _gurobi_status(model, GRB)
        has_solution = int(model.SolCount) > 0
        objective_bound = _finite_model_attribute(model, "ObjBound")
        barrier_iterations = int(round(float(model.BarIterCount)))
        simplex_iterations = float(model.IterCount)
        iterations = (
            barrier_iterations
            if barrier_iterations > 0
            else int(round(simplex_iterations))
        )
        result = _empty_result("gurobi", status, total_start)
        result.update(
            {
                "success": status == "optimal",
                "build_seconds": build_seconds,
                "solve_seconds": solve_seconds,
                "effective_tolerance": feasibility_tolerance,
                "iterations": iterations,
                "iteration_kind": (
                    "barrier"
                    if barrier_iterations > 0
                    else "simplex"
                ),
                "solver_details": {
                    "status_code": int(model.Status),
                    "solution_count": int(model.SolCount),
                    "simplex_iterations": simplex_iterations,
                    "barrier_iterations": barrier_iterations,
                    "runtime": float(model.Runtime),
                    "objective_bound": objective_bound,
                    "warm_start_used": warm_start_used,
                    "threads": settings["threads"],
                    "tolerance": tolerance,
                    "feasibility_tolerance": feasibility_tolerance,
                    "version": ".".join(
                        str(value) for value in gp.gurobi.version()
                    ),
                },
            }
        )
        if has_solution:
            postprocess_start = time.perf_counter()
            _complete_solution_result(
                result,
                prepared,
                np.asarray(x.X, dtype=np.float64),
                float(model.ObjVal),
                postprocess_start,
                domain_tolerance=feasibility_tolerance,
            )
        result["total_seconds"] = time.perf_counter() - total_start
        return result
    except Exception as error:
        kind = _failure_kind(error)
        result = _empty_result(
            "gurobi",
            "unavailable" if kind == "license" else "error",
            total_start,
            f"{type(error).__name__}: {error}",
        )
        result["solver_details"] = {"failure_kind": kind}
        return result
    finally:
        if model is not None:
            try:
                model.dispose()
            except Exception:
                pass
        if environment is not None:
            try:
                environment.dispose()
            except Exception:
                pass


__all__ = ["solve"]
