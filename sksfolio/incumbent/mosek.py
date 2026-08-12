"""Optional full-cardinality MOSEK reference backend.

The original sparse Markowitz problem is a mixed-integer convex quadratic
problem. Fusion uses the perspective-strengthened rotated-cone formulation
so that MOSEK solves the exact MISOCP rather than a continuous relaxation or
a QP approximation.
"""

from __future__ import annotations

import math
import sys
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ..relaxation._commercial_common import _failure_kind, _is_sparse_matrix
from ..relaxation.problem import MarkowitzInstance
from .evaluation import evaluate_incumbent
from .result import IncumbentResult
from .restricted_qp import solve_restricted_qp
from .state import IncumbentState
from .support import validate_branch_indices


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
    values = np.ascontiguousarray(
        np.asarray(matrix, dtype=np.float64)
    ).reshape(-1)
    return fusion.Matrix.dense(rows, columns, values)


def _warm_values(
    value: Any,
    dimension: int,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
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
        raise ValueError("MOSEK warm start has the wrong dimension")
    z = (
        (np.abs(x) > 1e-9).astype(float)
        if selectors is None
        else np.asarray(selectors, dtype=float).reshape(-1)
    )
    if z.shape != (dimension,):
        raise ValueError("MOSEK selector warm start has the wrong dimension")
    return x, z


def _info(model: Any, getter: str, name: str) -> Optional[Any]:
    try:
        value = getattr(model, getter)(name)
    except Exception:
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _version(mosek: Any) -> Optional[str]:
    try:
        return ".".join(str(value) for value in mosek.Env.getversion())
    except Exception:
        return None


def _unavailable(
    status: str,
    start: float,
    error: str,
    *,
    version: Optional[str] = None,
) -> IncumbentResult:
    elapsed = time.perf_counter() - start
    return IncumbentResult(
        {
            "status": status,
            "solver": "mosek",
            "formulation": "exact_binary_perspective_misocp",
            "x": None,
            "selectors": None,
            "upper_bound": None,
            "numerically_feasible": False,
            "error": error,
            "solve_seconds": elapsed,
            "total_seconds": elapsed,
            "version": version,
            "state": None,
        }
    )


def solve_mosek_incumbent(
    instance: MarkowitzInstance,
    *,
    warm_start: Optional[Any] = None,
    required_assets: Sequence[int] = (),
    forbidden_assets: Sequence[int] = (),
    time_limit: Optional[float] = None,
    options: Optional[Mapping[str, Any]] = None,
) -> IncumbentResult:
    """Solve the original binary sparse portfolio problem with MOSEK.

    The formulation uses binary selectors, ``0 <= x <= z``, the complete
    interval-row system, and rotated-cone epigraphs for factor risk and the
    ridge term.  Returned portfolios are independently checked before they
    are exposed as valid upper bounds.
    """
    start = time.perf_counter()
    try:
        import mosek
        import mosek.fusion as fusion
    except (ImportError, OSError) as error:
        raise ImportError(
            "the MOSEK Python package is required for "
            "solve_mosek_incumbent"
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
    verbose = bool(settings.pop("verbose", settings.pop("log", False)))
    threads = int(settings.pop("threads", settings.pop("numThreads", 0)))
    relative_gap = float(
        settings.pop("relative_gap", settings.pop("mioTolRelGap", 1e-4))
    )
    absolute_gap = float(
        settings.pop("absolute_gap", settings.pop("mioTolAbsGap", 0.0))
    )
    parameter_block = dict(settings.pop("mosek_params", {}))
    parameter_block.update(settings)
    if threads < 0:
        raise ValueError("threads must be nonnegative")
    if relative_gap < 0.0 or not math.isfinite(relative_gap):
        raise ValueError("relative_gap must be finite and nonnegative")
    if absolute_gap < 0.0 or not math.isfinite(absolute_gap):
        raise ValueError("absolute_gap must be finite and nonnegative")
    if time_limit is not None and (
        float(time_limit) <= 0.0 or not math.isfinite(float(time_limit))
    ):
        raise ValueError("time_limit must be positive and finite")

    model: Any = None
    try:
        model = fusion.Model("sksfolio_sparse_incumbent")
        if verbose:
            model.setLogHandler(sys.stdout)
        model.setSolverParam("log", int(verbose))
        model.setSolverParam("licenseWait", "off")
        model.setSolverParam("mioTolRelGap", relative_gap)
        model.setSolverParam("mioTolAbsGap", absolute_gap)
        if threads > 0:
            model.setSolverParam("numThreads", threads)
        if time_limit is not None:
            model.setSolverParam("mioMaxTime", float(time_limit))
        for name, value in parameter_block.items():
            model.setSolverParam(str(name), value)

        Domain = fusion.Domain
        Expr = fusion.Expr
        dimension = instance.dimension
        x = model.variable(
            "x",
            dimension,
            Domain.inRange(0.0, 1.0),
        )
        z = model.variable("z", dimension, Domain.binary())
        risk_epigraph = model.variable(
            "risk_epigraph",
            1,
            Domain.greaterThan(0.0),
        )
        perspective_epigraph = model.variable(
            "perspective_epigraph",
            dimension,
            Domain.greaterThan(0.0),
        )

        model.constraint(
            "activation",
            Expr.sub(x, z),
            Domain.lessThan(0.0),
        )
        model.constraint(
            "cardinality",
            Expr.sum(z),
            Domain.lessThan(float(instance.k)),
        )
        if required.size:
            model.constraint(
                "required",
                z.pick(required.tolist()),
                Domain.equalsTo(1.0),
            )
        if forbidden.size:
            model.constraint(
                "forbidden",
                z.pick(forbidden.tolist()),
                Domain.equalsTo(0.0),
            )

        factor_matrix = _fusion_matrix(fusion, instance.B.T)
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
                Expr.mul(0.5, perspective_epigraph),
                z,
                x,
            ),
            Domain.inRotatedQCone().axis(1),
        )
        if instance.rows:
            lower = np.asarray(instance.lower, dtype=float)
            upper = np.asarray(instance.upper, dtype=float)
            finite_lower = np.flatnonzero(np.isfinite(lower))
            finite_upper = np.flatnonzero(np.isfinite(upper))
            if finite_lower.size:
                lower_matrix = _fusion_matrix(
                    fusion,
                    instance.C[finite_lower, :],
                )
                model.constraint(
                    "portfolio_lower",
                    Expr.mul(lower_matrix, x),
                    Domain.greaterThan(lower[finite_lower]),
                )
            if finite_upper.size:
                upper_matrix = _fusion_matrix(
                    fusion,
                    instance.C[finite_upper, :],
                )
                model.constraint(
                    "portfolio_upper",
                    Expr.mul(upper_matrix, x),
                    Domain.lessThan(upper[finite_upper]),
                )

        model.objective(
            "objective",
            fusion.ObjectiveSense.Minimize,
            Expr.add(
                [
                    Expr.mul(0.5, risk_epigraph.index(0)),
                    Expr.mul(
                        0.5 * float(instance.perspective_weight),
                        Expr.sum(perspective_epigraph),
                    ),
                    Expr.mul(
                        -float(instance.return_reward),
                        Expr.dot(
                            np.asarray(instance.mu, dtype=float),
                            x,
                        ),
                    ),
                ]
            ),
        )

        warm_x, warm_z = _warm_values(warm_start, dimension)
        warm_start_used = warm_x is not None and warm_z is not None
        if warm_start_used:
            clipped_x = np.clip(warm_x, 0.0, 1.0)
            rounded_z = np.clip(np.rint(warm_z), 0.0, 1.0)
            x.setLevel(clipped_x)
            z.setLevel(rounded_z)
            factor = np.asarray(instance.B.T @ clipped_x).reshape(-1)
            risk_epigraph.setLevel(
                np.asarray([float(factor @ factor)])
            )
            perspective_epigraph.setLevel(
                np.divide(
                    clipped_x * clipped_x,
                    rounded_z,
                    out=np.zeros_like(clipped_x),
                    where=rounded_z > 0.5,
                )
            )

        model.acceptedSolutionStatus(fusion.AccSolutionStatus.Anything)
        build_seconds = time.perf_counter() - start
        solve_start = time.perf_counter()
        model.solve()
        solve_seconds = time.perf_counter() - solve_start

        primal_status = model.getPrimalSolutionStatus()
        problem_status = model.getProblemStatus()
        solution_count = _info(
            model,
            "getSolverIntInfo",
            "mioNumIntSolutions",
        )
        has_solution = bool(solution_count and int(solution_count) > 0)
        objective_bound = None
        bound_defined = _info(
            model,
            "getSolverIntInfo",
            "mioObjBoundDefined",
        )
        if bound_defined:
            objective_bound = _info(
                model,
                "getSolverDoubleInfo",
                "mioObjBound",
            )
        if primal_status == fusion.SolutionStatus.Optimal:
            status = "optimal"
        elif has_solution:
            status = "feasible"
            if time_limit is not None:
                mio_time = _info(
                    model,
                    "getSolverDoubleInfo",
                    "mioTime",
                )
                if mio_time is not None and mio_time >= 0.98 * float(
                    time_limit
                ):
                    status = "time_limit"
        else:
            problem_text = str(problem_status).lower()
            status = (
                "infeasible"
                if "primalinfeasible" in problem_text
                else "no_solution"
            )

        common = {
            "status": status,
            "solver": "mosek",
            "formulation": "exact_binary_perspective_misocp",
            "solver_objective_bound": objective_bound,
            "solver_mip_gap": _info(
                model,
                "getSolverDoubleInfo",
                "mioObjRelGap",
            ),
            "node_count": _info(
                model,
                "getSolverIntInfo",
                "mioNumSolvedNodes",
            ),
            "solution_count": solution_count,
            "build_seconds": build_seconds,
            "solve_seconds": solve_seconds,
            "warm_start_used": warm_start_used,
            "problem_status": str(problem_status),
            "primal_solution_status": str(primal_status),
            "version": _version(mosek),
        }
        if not has_solution:
            common["total_seconds"] = time.perf_counter() - start
            return IncumbentResult(
                {
                    **common,
                    "x": None,
                    "selectors": None,
                    "upper_bound": None,
                    "numerically_feasible": False,
                    "state": None,
                }
            )

        weights = np.asarray(x.level(), dtype=float)
        selectors = np.clip(
            np.rint(np.asarray(z.level(), dtype=float)),
            0.0,
            1.0,
        )
        raw_incumbent_objective = float(model.primalObjValue())
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
        common["total_seconds"] = time.perf_counter() - start
        return IncumbentResult(
            {
                **common,
                "x": weights,
                "selectors": selectors,
                "upper_bound": upper_bound,
                "numerically_feasible": bool(
                    diagnostics["numerically_feasible"]
                ),
                "floating_point_certified": False,
                "solver_incumbent_objective": raw_incumbent_objective,
                "incumbent_polish_status": polish_status,
                "incumbent_polish_seconds": polish_seconds,
                "diagnostics": diagnostics,
                "state": state,
            }
        )
    except Exception as error:
        kind = _failure_kind(error)
        return _unavailable(
            "unavailable" if kind == "license" else "error",
            start,
            f"{type(error).__name__}: {error}",
            version=_version(mosek),
        )
    finally:
        if model is not None:
            try:
                model.dispose()
            except Exception:
                pass


__all__ = ["solve_mosek_incumbent"]
