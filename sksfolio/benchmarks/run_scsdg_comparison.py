"""Compare FISTA, PDHG, and SC-SDG at a common safe-bound target."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from .bertsimas_cory_wright import (
    BCWCase,
    CONSTRAINT_PROFILES,
    HISTORICAL_UNIVERSES,
    generate_synthetic_case,
)
from ..relaxation import solve_relaxation


METHODS = ("fista", "pdhg", "scsdg")


def _case(args: argparse.Namespace) -> BCWCase:
    _, dimension, ranks = HISTORICAL_UNIVERSES[args.universe]
    if args.rank not in ranks:
        raise ValueError(
            f"rank must be one of {ranks} for {args.universe}"
        )
    return BCWCase(
        family="historical",
        universe=args.universe,
        dimension=dimension,
        rank=args.rank,
        k=args.k,
        gamma_scale=args.gamma_scale,
        regime="unconstrained",
    )


def _solver_options(
    method: str,
    args: argparse.Namespace,
    cutoff: float,
) -> Dict[str, Any]:
    common = {
        "threads": args.threads,
        "max_iterations": args.max_iterations,
        "time_limit": args.time_limit,
        "tolerance": args.solver_tolerance,
        "feasibility_tolerance": args.solver_tolerance,
        "dual_bound_cutoff": cutoff,
    }
    if method == "fista":
        return {
            **common,
            "history_interval": args.check_interval,
            "prox_tolerance": args.prox_tolerance,
            "prox_max_iterations": args.prox_max_iterations,
            "prox_oracle": "auto",
            "restart_strategy": "gradient",
        }
    if method == "pdhg":
        return {
            **common,
            "check_interval": args.check_interval,
            "min_epoch": max(2 * args.check_interval, 50),
            "max_epoch": 2_000,
        }
    return {
        **common,
        "check_interval": args.check_interval,
        "restart_check_interval": args.scsdg_restart_check_interval,
        "theta_parameter": 2.0,
        "continuation_offset": args.scsdg_continuation_offset,
        "target_cbar": args.scsdg_cbar,
        "step_ratio": args.scsdg_step_ratio,
        "constraint_dual_weight": args.scsdg_constraint_weight,
        "restart": not args.scsdg_no_restart,
        "line_search": args.scsdg_line_search,
        "line_search_mode": args.scsdg_line_search_mode,
        "line_search_auto_row_threshold": (
            args.scsdg_line_search_auto_row_threshold
        ),
        "line_search_initial_scale": (
            args.scsdg_line_search_initial_scale
        ),
        "line_search_growth": args.scsdg_line_search_growth,
        "line_search_shrink": args.scsdg_line_search_shrink,
        "line_search_max_scale": args.scsdg_line_search_max_scale,
        "line_search_safety": args.scsdg_line_search_safety,
    }


def _summary_row(
    method: str,
    result: Any,
    optimum: float,
    target: float,
    problem: Any,
) -> Dict[str, Any]:
    scale = max(1.0, abs(optimum))
    certificate = result.dual_certificate
    diagnostics = result.raw.get("diagnostics", {})
    violations = diagnostics.get("violations", {})
    return {
        "method": method,
        "status": result.status,
        "dimension": problem.dimension,
        "rank": problem.rank,
        "rows": problem.rows,
        "k": problem.k,
        "safe_gap_target": target,
        "reference_objective": optimum,
        "objective": result.objective,
        "safe_dual_bound": result.safe_dual_bound,
        "safe_dual_relative_error": (
            (optimum - result.safe_dual_bound) / scale
            if result.safe_dual_bound is not None
            else None
        ),
        "certificate_verified": (
            certificate.verify(problem)
            if certificate is not None
            else False
        ),
        "maximum_violation": violations.get("maximum"),
        "iterations": result.raw.get("iterations"),
        "pava_calls": result.raw.get("pava_calls"),
        "restarts": result.raw.get("restarts"),
        "smoothed_gap_evaluations": result.raw.get(
            "smoothed_gap_evaluations"
        ),
        "line_search_enabled": result.raw.get(
            "line_search_enabled"
        ),
        "line_search_mode": result.raw.get("line_search_mode"),
        "line_search_mode_resolved": result.raw.get(
            "line_search_mode_resolved"
        ),
        "line_search_trials": result.raw.get("line_search_trials"),
        "line_search_backtracks": result.raw.get(
            "line_search_backtracks"
        ),
        "line_search_final_scale": result.raw.get(
            "line_search_final_scale"
        ),
        "line_search_average_accepted_scale": result.raw.get(
            "line_search_average_accepted_scale"
        ),
        "solve_seconds": result.raw.get("solve_seconds"),
        "setup_seconds": result.raw.get("setup_seconds"),
        "end_to_end_seconds": result.raw.get(
            "end_to_end_seconds"
        ),
    }


def _history_rows(
    method: str,
    result: Any,
    optimum: float,
) -> list[Dict[str, Any]]:
    scale = max(1.0, abs(optimum))
    rows = []
    for point in result.raw.get("history", []):
        bound = point.get(
            "best_dual_bound",
            point.get("best_dual_lower_bound"),
        )
        rows.append(
            {
                "method": method,
                "iteration": point.get("iteration"),
                "elapsed_seconds": point.get("elapsed_seconds"),
                "best_dual_bound": bound,
                "safe_dual_relative_error": (
                    max(optimum - float(bound), 0.0) / scale
                    if bound is not None
                    else None
                ),
                "residual": point.get("residual"),
                "relative_residual": point.get("relative_residual"),
                "violation": point.get("violation"),
                "smoothed_gap": point.get("smoothed_gap"),
                "relative_smoothed_gap": point.get(
                    "relative_smoothed_gap"
                ),
                "restart": point.get("restart"),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, rows: list[Dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    for method in METHODS:
        selected = [
            row
            for row in rows
            if row["method"] == method
            and row["elapsed_seconds"] is not None
            and row["safe_dual_relative_error"] is not None
        ]
        if not selected:
            continue
        times = np.asarray(
            [float(row["elapsed_seconds"]) for row in selected]
        )
        errors = np.maximum(
            np.asarray(
                [
                    float(row["safe_dual_relative_error"])
                    for row in selected
                ]
            ),
            1e-16,
        )
        axis.semilogy(times, errors, label=method.upper(), linewidth=2)
    axis.set_xlabel("solve time (seconds)")
    axis.set_ylabel("safe dual-bound error")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Compare FISTA, PDHG, and restarted SC-SDG using the same "
            "Gurobi reference and safe-dual stopping target"
        )
    )
    result.add_argument(
        "--output",
        type=Path,
        default=Path("scsdg_comparison.csv"),
    )
    result.add_argument(
        "--history-output",
        type=Path,
        default=Path("scsdg_comparison_history.csv"),
    )
    result.add_argument(
        "--plot",
        type=Path,
        default=Path("scsdg_comparison.png"),
    )
    result.add_argument(
        "--universe",
        choices=tuple(HISTORICAL_UNIVERSES),
        default="sp500",
    )
    result.add_argument("--rank", type=int, default=50)
    result.add_argument("--k", type=int, default=10)
    result.add_argument(
        "--gamma-scale",
        type=float,
        choices=(1.0, 100.0),
        default=100.0,
    )
    result.add_argument(
        "--constraint-profile",
        choices=tuple(CONSTRAINT_PROFILES),
        default="bcw",
    )
    result.add_argument("--seed", type=int, default=7)
    result.add_argument("--target-iterations", type=int, default=200)
    result.add_argument("--safe-gap-target", type=float, default=1e-6)
    result.add_argument("--solver-tolerance", type=float, default=1e-10)
    result.add_argument("--prox-tolerance", type=float, default=1e-9)
    result.add_argument("--prox-max-iterations", type=int, default=2_000)
    result.add_argument("--max-iterations", type=int, default=200_000)
    result.add_argument("--time-limit", type=float, default=60.0)
    result.add_argument("--threads", type=int, default=1)
    result.add_argument("--check-interval", type=int, default=25)
    result.add_argument("--scsdg-cbar", type=float, default=0.1)
    result.add_argument("--scsdg-step-ratio", type=float, default=1.0)
    result.add_argument(
        "--scsdg-constraint-weight",
        default="auto",
    )
    result.add_argument(
        "--scsdg-continuation-offset",
        type=float,
        default=3.0,
    )
    result.add_argument(
        "--scsdg-restart-check-interval",
        type=int,
        default=10,
    )
    result.add_argument(
        "--scsdg-no-restart",
        action="store_true",
    )
    result.add_argument(
        "--scsdg-line-search",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    result.add_argument(
        "--scsdg-line-search-mode",
        choices=("auto", "operator", "majorization"),
        default="auto",
    )
    result.add_argument(
        "--scsdg-line-search-auto-row-threshold",
        type=int,
        default=16,
    )
    result.add_argument(
        "--scsdg-line-search-initial-scale",
        type=float,
        default=1.0,
    )
    result.add_argument(
        "--scsdg-line-search-growth",
        type=float,
        default=1.1,
    )
    result.add_argument(
        "--scsdg-line-search-shrink",
        type=float,
        default=0.5,
    )
    result.add_argument(
        "--scsdg-line-search-max-scale",
        type=float,
        default=1024.0,
    )
    result.add_argument(
        "--scsdg-line-search-safety",
        type=float,
        default=0.99,
    )
    return result


def main(arguments: Sequence[str] | None = None) -> None:
    args = parser().parse_args(arguments)
    if (
        not math.isfinite(args.safe_gap_target)
        or args.safe_gap_target <= 0.0
    ):
        raise ValueError("safe-gap-target must be positive and finite")
    case = _case(args)
    problem = generate_synthetic_case(
        case,
        seed=args.seed,
        target_iterations=args.target_iterations,
        constraint_profile=args.constraint_profile,
    )
    reference = solve_relaxation(
        problem,
        "gurobi.python",
        options={
            "threads": args.threads,
            "tolerance": min(args.solver_tolerance, 1e-10),
            "time_limit": args.time_limit,
            "log": False,
        },
    )
    if reference.status != "optimal" or reference.objective is None:
        raise RuntimeError(
            "an optimal Gurobi reference is required for this benchmark"
        )
    optimum = float(reference.objective)
    cutoff = (
        optimum
        - args.safe_gap_target * max(1.0, abs(optimum))
    )

    summaries = []
    histories = []
    for method in METHODS:
        result = solve_relaxation(
            problem,
            method,
            variant="metric-linesearch-restart",
            pava="partial_sort",
            options=_solver_options(method, args, cutoff),
        )
        summaries.append(
            _summary_row(
                method,
                result,
                optimum,
                args.safe_gap_target,
                problem,
            )
        )
        histories.extend(_history_rows(method, result, optimum))

    _write_csv(args.output, summaries)
    _write_csv(args.history_output, histories)
    if args.plot is not None:
        _plot(args.plot, histories)
    for row in summaries:
        print(
            row["method"],
            row["status"],
            f"{row['end_to_end_seconds']:.6g}s",
            f"safe_error={row['safe_dual_relative_error']:.3e}",
        )


if __name__ == "__main__":
    main()
