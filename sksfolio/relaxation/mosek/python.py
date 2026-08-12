"""Native MOSEK Fusion implementation of the perspective relaxation.

The optional `mosek` package is imported only when :func:`solve` is
called. Missing packages and licenses are returned as normalized unavailable
results.
"""

from __future__ import annotations

import sys
import time
from typing import Any, Dict, Optional

import numpy as np

from .._commercial_common import (
    _complete_solution_result,
    _empty_result,
    _failure_kind,
    _is_sparse_matrix,
    _prepare_instance,
    _prepare_options,
    _warm_start_values,
)


def _fusion_matrix(fusion: Any, matrix: Any) -> Any:
    rows, columns = int(matrix.shape[0]), int(matrix.shape[1])
    if _is_sparse_matrix(matrix):
        coo = matrix.tocoo()
        return fusion.Matrix.sparse(
            rows,
            columns,
            np.asarray(coo.row, dtype=np.int32),
            np.asarray(coo.col, dtype=np.int32),
            np.asarray(coo.data, dtype=np.float64),
        )
    flat = np.ascontiguousarray(
        np.asarray(matrix, dtype=np.float64)
    ).reshape(-1)
    return fusion.Matrix.dense(rows, columns, flat)


def _mosek_info(
    model: Any,
    getter: str,
    name: str,
) -> Optional[Any]:
    try:
        value = getattr(model, getter)(name)
    except Exception:
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value


def _mosek_status(
    fusion: Any,
    primal_status: Any,
    problem_status: Any,
) -> str:
    if primal_status == fusion.SolutionStatus.Optimal:
        return "optimal"
    if primal_status == fusion.SolutionStatus.Feasible:
        return "feasible"
    problem_text = str(problem_status).lower()
    if "primalinfeasibleorunbounded" in problem_text:
        return "infeasible_or_unbounded"
    if "primalinfeasible" in problem_text:
        return "infeasible"
    if "dualinfeasible" in problem_text:
        return "unbounded"
    if "illposed" in problem_text:
        return "numerical_error"
    return "unknown"


def _mosek_version(mosek: Any) -> Optional[str]:
    try:
        return ".".join(str(value) for value in mosek.Env.getversion())
    except Exception:
        return None


def solve(
    instance: Any,
    options: Optional[Any] = None,
) -> Dict[str, Any]:
    """Solve an instance with native ``mosek.fusion``.

    Supported options are ``threads``, ``tolerance``, ``time_limit``, ``log``,
    ``warm_start``, and ``mosek_params``.  The last item is a mapping of
    additional Fusion solver parameter names to values.
    """
    total_start = time.perf_counter()
    try:
        prepared = _prepare_instance(instance)
        settings = _prepare_options(options, "mosek_params")
    except Exception as error:
        return _empty_result(
            "mosek",
            "error",
            total_start,
            f"{type(error).__name__}: {error}",
        )

    try:
        import mosek
        import mosek.fusion as fusion
    except (ImportError, OSError) as error:
        return _empty_result(
            "mosek",
            "unavailable",
            total_start,
            f"MOSEK Fusion is unavailable: {type(error).__name__}: {error}",
        )

    model: Any = None
    try:
        model = fusion.Model("markowitz_perspective")
        if settings["log"]:
            model.setLogHandler(sys.stdout)
        model.setSolverParam("log", int(settings["log"]))
        model.setSolverParam("licenseWait", "off")
        # MOSEK 11.2 rejects the generic interior-point selector for a
        # conic model (error 1550).  Select its conic optimizer explicitly.
        model.setSolverParam("optimizer", "conic")
        model.setSolverParam("intpntBasis", "never")
        if settings["threads"] > 0:
            model.setSolverParam("numThreads", settings["threads"])
        tolerance = settings["tolerance"]
        feasibility_tolerance = max(1e-9, min(1e-2, tolerance))
        model.setSolverParam("intpntCoTolRelGap", tolerance)
        model.setSolverParam(
            "intpntCoTolPfeas",
            feasibility_tolerance,
        )
        model.setSolverParam(
            "intpntCoTolDfeas",
            feasibility_tolerance,
        )
        if settings["time_limit"] is not None:
            model.setSolverParam(
                "optimizerMaxTime",
                settings["time_limit"],
            )
        for name, value in settings["solver_params"].items():
            model.setSolverParam(str(name), value)

        Domain = fusion.Domain
        Expr = fusion.Expr
        ObjectiveSense = fusion.ObjectiveSense

        x = model.variable(
            "x",
            prepared.dimension,
            Domain.inRange(0.0, 1.0),
        )
        z = model.variable(
            "z",
            prepared.dimension,
            Domain.inRange(0.0, 1.0),
        )
        t = model.variable(
            "t",
            prepared.dimension,
            Domain.greaterThan(0.0),
        )
        risk_epigraph = model.variable(
            "risk_epigraph",
            1,
            Domain.greaterThan(0.0),
        )

        model.constraint(
            "long_only_link",
            Expr.sub(x, z),
            Domain.lessThan(0.0),
        )
        model.constraint(
            "perspective_budget",
            Expr.sum(z),
            Domain.lessThan(float(prepared.k)),
        )

        factor_matrix = _fusion_matrix(fusion, prepared.B.T)
        model.constraint(
            "factor_risk",
            Expr.vstack(
                risk_epigraph.index(0),
                0.5,
                Expr.mul(factor_matrix, x),
            ),
            Domain.inRotatedQCone(),
        )
        model.constraint(
            "perspective_cones",
            Expr.hstack(
                Expr.mul(0.5, t),
                z,
                x,
            ),
            Domain.inRotatedQCone().axis(1),
        )
        if prepared.constraints:
            finite_lower = np.flatnonzero(np.isfinite(prepared.lower))
            finite_upper = np.flatnonzero(np.isfinite(prepared.upper))
            if finite_lower.size:
                lower_matrix = _fusion_matrix(
                    fusion,
                    prepared.C[finite_lower, :],
                )
                model.constraint(
                    "portfolio_lower",
                    Expr.mul(lower_matrix, x),
                    Domain.greaterThan(prepared.lower[finite_lower]),
                )
            if finite_upper.size:
                upper_matrix = _fusion_matrix(
                    fusion,
                    prepared.C[finite_upper, :],
                )
                model.constraint(
                    "portfolio_upper",
                    Expr.mul(upper_matrix, x),
                    Domain.lessThan(prepared.upper[finite_upper]),
                )

        model.objective(
            "objective",
            ObjectiveSense.Minimize,
            Expr.add(
                [
                    Expr.mul(0.5, risk_epigraph.index(0)),
                    Expr.mul(
                        0.5 * prepared.perspective_weight,
                        Expr.sum(t),
                    ),
                    Expr.mul(
                        -prepared.return_reward,
                        Expr.dot(prepared.mu, x),
                    ),
                ]
            ),
        )

        warm_start_used = False
        if settings["warm_start"]:
            start_values = _warm_start_values(
                prepared,
                settings.get("warm_start_state"),
            )
            if start_values is not None:
                (
                    x_start,
                    z_start,
                    t_start,
                    _,
                    risk_start,
                ) = start_values
                x.setLevel(x_start)
                z.setLevel(z_start)
                t.setLevel(t_start)
                risk_epigraph.setLevel(np.asarray([risk_start]))
                warm_start_used = True

        model.acceptedSolutionStatus(
            fusion.AccSolutionStatus.Feasible
        )
        build_seconds = time.perf_counter() - total_start
        solve_start = time.perf_counter()
        model.solve()
        solve_seconds = time.perf_counter() - solve_start

        primal_status = model.getPrimalSolutionStatus()
        dual_status = model.getDualSolutionStatus()
        problem_status = model.getProblemStatus()
        status = _mosek_status(fusion, primal_status, problem_status)
        has_solution = primal_status in (
            fusion.SolutionStatus.Optimal,
            fusion.SolutionStatus.Feasible,
        )
        dual_objective = None
        if dual_status in (
            fusion.SolutionStatus.Optimal,
            fusion.SolutionStatus.Feasible,
        ):
            try:
                candidate = float(model.dualObjValue())
                if np.isfinite(candidate):
                    dual_objective = candidate
            except Exception:
                pass
        result = _empty_result("mosek", status, total_start)
        result.update(
            {
                "success": status == "optimal",
                "build_seconds": build_seconds,
                "solve_seconds": solve_seconds,
                "effective_tolerance": feasibility_tolerance,
                "iterations": _mosek_info(
                    model,
                    "getSolverIntInfo",
                    "intpntIter",
                ),
                "solver_details": {
                    "primal_solution_status": str(primal_status),
                    "dual_solution_status": str(dual_status),
                    "problem_status": str(problem_status),
                    "objective_bound": dual_objective,
                    "optimizer_seconds": _mosek_info(
                        model,
                        "getSolverDoubleInfo",
                        "optimizerTime",
                    ),
                    "barrier_iterations": _mosek_info(
                        model,
                        "getSolverIntInfo",
                        "intpntIter",
                    ),
                    "warm_start_used": warm_start_used,
                    "threads": settings["threads"],
                    "tolerance": tolerance,
                    "feasibility_tolerance": feasibility_tolerance,
                    "version": _mosek_version(mosek),
                },
            }
        )
        if has_solution:
            postprocess_start = time.perf_counter()
            _complete_solution_result(
                result,
                prepared,
                np.asarray(x.level(), dtype=np.float64),
                float(model.primalObjValue()),
                postprocess_start,
                domain_tolerance=feasibility_tolerance,
            )
        result["total_seconds"] = time.perf_counter() - total_start
        return result
    except Exception as error:
        kind = _failure_kind(error)
        result = _empty_result(
            "mosek",
            "unavailable" if kind == "license" else "error",
            total_start,
            f"{type(error).__name__}: {error}",
        )
        result["solver_details"] = {
            "failure_kind": kind,
            "version": _mosek_version(mosek),
        }
        return result
    finally:
        if model is not None:
            try:
                model.dispose()
            except Exception:
                pass


__all__ = ["solve"]
