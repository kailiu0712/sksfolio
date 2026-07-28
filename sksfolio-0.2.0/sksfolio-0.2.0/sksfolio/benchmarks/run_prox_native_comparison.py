"""Benchmark the constrained prox with Python PAVA, native PAVA, and Gurobi."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from .bertsimas_cory_wright import BCWCase, generate_synthetic_case
from ..relaxation.fista import LinearConstraintProx
from ..relaxation.fista.solver import _estimate_lipschitz
from ..relaxation.pdhg.pava import prox_python_partial_sort


class _PythonPartialSortProx(LinearConstraintProx):
    """Retain the NumPy PAVA implementation for an in-process A/B test."""

    def _pava(self, argument: np.ndarray, gamma: float) -> np.ndarray:
        return prox_python_partial_sort(argument, gamma, self.k)


def _representative_prox_data() -> tuple[Any, np.ndarray, float]:
    case = BCWCase(
        family="historical",
        universe="sp500",
        dimension=499,
        rank=50,
        k=10,
        gamma_scale=100.0,
        regime="constrained",
    )
    problem = generate_synthetic_case(
        case,
        seed=7,
        target_iterations=200,
        constraint_profile="many",
    )
    lipschitz, _ = _estimate_lipschitz(
        problem.B,
        problem.dimension,
        problem.rank,
        20,
        problem.perspective_weight,
    )
    anchor = np.asarray(problem.anchor, dtype=float)
    gradient = (
        problem.B @ (problem.B.T @ anchor)
        - problem.return_reward * np.asarray(problem.mu, dtype=float)
    )
    argument = anchor - np.asarray(gradient, dtype=float) / lipschitz
    gamma = float(problem.perspective_weight) / lipschitz
    return problem, argument, gamma


def _first_order_rows(
    oracle_type: type[LinearConstraintProx],
    label: str,
    problem: Any,
    argument: np.ndarray,
    gamma: float,
    repeats: int,
    tolerance: float,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    build_start = time.perf_counter()
    oracle = oracle_type(
        problem.C,
        problem.lower,
        problem.upper,
        problem.k,
        pava_method="partial_sort",
        tolerance=tolerance,
        max_iterations=1000,
        adaptive_restart=True,
    )
    build_seconds = time.perf_counter() - build_start
    rows: list[dict[str, Any]] = []
    point = np.empty(problem.dimension)
    for repetition in range(repeats):
        oracle.reset()
        start = time.perf_counter()
        result = oracle.solve(argument, gamma)
        elapsed = time.perf_counter() - start
        point = result.x
        rows.append(
            {
                "solver": label,
                "repetition": repetition + 1,
                "status": (
                    "converged" if result.converged else "iteration_limit"
                ),
                "build_seconds": build_seconds if repetition == 0 else 0.0,
                "solve_seconds": elapsed,
                "total_seconds": elapsed,
                "iterations": result.iterations,
                "pava_calls": result.pava_calls,
                "constraint_violation": result.constraint_violation,
                "maximum_error_to_gurobi": None,
            }
        )
    return rows, point


def _gurobi_row(
    problem: Any,
    argument: np.ndarray,
    gamma: float,
    repetition: int,
    tolerance: float,
) -> tuple[dict[str, Any], np.ndarray]:
    import gurobipy as gp

    total_start = time.perf_counter()
    environment = gp.Env(empty=True)
    environment.setParam("OutputFlag", 0)
    environment.start()
    model = gp.Model("constrained_perspective_prox", env=environment)
    try:
        model.Params.OutputFlag = 0
        model.Params.NonConvex = 0
        model.Params.Method = 2
        model.Params.Crossover = 0
        model.Params.Threads = 1
        model.Params.FeasibilityTol = max(1e-9, tolerance)
        model.Params.OptimalityTol = max(1e-9, tolerance)
        model.Params.BarConvTol = tolerance
        model.Params.BarQCPConvTol = tolerance

        dimension = problem.dimension
        x = model.addMVar(dimension, lb=0.0, ub=1.0, name="x")
        z = model.addMVar(dimension, lb=0.0, ub=1.0, name="z")
        perspective = model.addMVar(dimension, lb=0.0, name="t")
        model.addConstr(x <= z)
        model.addConstr(z.sum() <= problem.k)
        model.addConstr(x * x <= perspective * z)

        row_value = problem.C @ x
        lower = np.asarray(problem.lower, dtype=float)
        upper = np.asarray(problem.upper, dtype=float)
        finite_lower = np.isfinite(lower)
        finite_upper = np.isfinite(upper)
        if np.any(finite_lower):
            model.addConstr(row_value[finite_lower] >= lower[finite_lower])
        if np.any(finite_upper):
            model.addConstr(row_value[finite_upper] <= upper[finite_upper])

        difference = x - argument
        model.setObjective(
            0.5 * (difference @ difference)
            + 0.5 * gamma * perspective.sum(),
            gp.GRB.MINIMIZE,
        )
        model.update()
        build_seconds = time.perf_counter() - total_start
        solve_start = time.perf_counter()
        model.optimize()
        solve_seconds = time.perf_counter() - solve_start
        if model.Status != gp.GRB.OPTIMAL:
            raise RuntimeError(f"Gurobi status {model.Status}")
        point = np.asarray(x.X, dtype=float)
        values = np.asarray(problem.C @ point, dtype=float).reshape(-1)
        violation = max(
            0.0,
            float(np.max(lower[finite_lower] - values[finite_lower])),
            float(np.max(values[finite_upper] - upper[finite_upper])),
        )
        row = {
            "solver": "gurobi",
            "repetition": repetition,
            "status": "optimal",
            "build_seconds": build_seconds,
            "solve_seconds": solve_seconds,
            "total_seconds": time.perf_counter() - total_start,
            "iterations": int(round(float(model.BarIterCount))),
            "pava_calls": None,
            "constraint_violation": violation,
            "maximum_error_to_gurobi": 0.0,
        }
        return row, point
    finally:
        model.dispose()
        environment.dispose()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare one 262-row constrained prox using the NumPy and "
            "compiled partial-sort PAVA kernels and an exact Gurobi QCP"
        )
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument(
        "--reference-tolerance",
        type=float,
        default=1e-10,
        help="Gurobi QCP tolerance used for the high-accuracy reference",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--skip-gurobi",
        action="store_true",
        help="run only the two custom implementations",
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(arguments)
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    problem, argument, gamma = _representative_prox_data()
    python_rows, python_point = _first_order_rows(
        _PythonPartialSortProx,
        "dual_fista_python_pava",
        problem,
        argument,
        gamma,
        args.repeats,
        args.tolerance,
    )
    native_rows, native_point = _first_order_rows(
        LinearConstraintProx,
        "dual_fista_native_pava",
        problem,
        argument,
        gamma,
        args.repeats,
        args.tolerance,
    )
    rows = python_rows + native_rows
    if not args.skip_gurobi:
        gurobi_rows: list[dict[str, Any]] = []
        gurobi_point = np.empty(problem.dimension)
        for repetition in range(1, args.repeats + 1):
            row, gurobi_point = _gurobi_row(
                problem,
                argument,
                gamma,
                repetition,
                args.reference_tolerance,
            )
            gurobi_rows.append(row)
        for row in python_rows:
            row["maximum_error_to_gurobi"] = float(
                np.max(np.abs(python_point - gurobi_point))
            )
        for row in native_rows:
            row["maximum_error_to_gurobi"] = float(
                np.max(np.abs(native_point - gurobi_point))
            )
        rows.extend(gurobi_rows)

    fields = list(rows[0])
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(output)
    for row in rows:
        print(
            f"{row['solver']:26s} "
            f"{row['solve_seconds'] * 1e3:9.3f} ms "
            f"{row['status']}"
        )


if __name__ == "__main__":
    main()
