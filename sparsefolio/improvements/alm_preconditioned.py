"""Column-preconditioned ALM polishing with an unchanged portfolio model."""
from __future__ import annotations

import time

import numpy as np

from ..core import Problem, evaluate
from .alm_polished import solve as solve_polished


def transform_problem(problem: Problem) -> tuple[Problem, np.ndarray]:
    """Substitute x = D w, using clipped exposure-column norm equilibration."""
    squared_norms = np.sum(problem.K * problem.K, axis=0)
    if not np.all(np.isfinite(squared_norms)) or np.any(squared_norms <= 0):
        raise ValueError("Every exposure column must have a finite positive norm")
    diagonal = np.clip(np.sqrt(np.median(squared_norms) / squared_norms), .5, 2.0)
    scaled = Problem(
        B=np.ascontiguousarray(problem.B * diagonal),
        c=problem.c * diagonal,
        gamma=problem.gamma * diagonal * diagonal,
        u=problem.u / diagonal,
        k=problem.k,
        K=np.ascontiguousarray(problem.K * diagonal),
        lo=problem.lo, hi=problem.hi,
        raw_K=(np.ascontiguousarray(problem.raw_K * diagonal)
               if problem.raw_K is not None else None),
        raw_lo=problem.raw_lo, raw_hi=problem.raw_hi,
        row_scale=problem.row_scale,
        anchor=(problem.anchor / diagonal if problem.anchor is not None else None),
        metadata=problem.metadata,
    )
    return scaled, diagonal


def solve(problem, mode="alm_preconditioned", time_limit=15.0, tol=1e-6,
          feas_tol=1e-7, max_iter=100):
    """Run alm_polished in scaled coordinates and certify original coordinates."""
    if mode != "alm_preconditioned":
        raise ValueError("mode must be alm_preconditioned")
    start = time.perf_counter()
    scaled, diagonal = transform_problem(problem)
    preprocessing_seconds = time.perf_counter() - start
    inner = solve_polished(scaled, time_limit=max(1e-9, time_limit-preprocessing_seconds),
                           tol=tol, feas_tol=feas_tol, max_iter=max_iter)
    inner["x"] = diagonal * inner["x"]
    inner["metrics"] = evaluate(problem, inner["x"], inner["y"])
    inner["seconds"] = time.perf_counter() - start
    solved = (inner["metrics"]["rel_gap"] <= tol
              and inner["metrics"]["feasibility"] <= feas_tol)
    inner["status"] = ("converged" if solved else
                       "time_limit" if inner["seconds"] >= time_limit else "not_converged")
    inner["diagnostics"].update(mode=mode, seconds=inner["seconds"],
                                preprocessing_seconds=preprocessing_seconds,
                                column_scale_min=float(diagonal.min()),
                                column_scale_max=float(diagonal.max()))
    return inner
