"""Matched safe-screened BnB, Gurobi, and MOSEK benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Optional, Sequence

from ..bnb import solve_bnb
from ..incumbent import (
    solve_gurobi_incumbent,
    solve_incumbent,
    solve_mosek_incumbent,
)
from ..relaxation import load_instance_bundle, solve_relaxation
from .instance_generator import generate_instance


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Compare certificate-driven BnB with Gurobi and MOSEK on "
            "the same constrained sparse Markowitz instance"
        )
    )
    result.add_argument("--bundle", type=Path, default=None)
    result.add_argument("--output", type=Path, default=None)
    result.add_argument("--dimension", type=int, default=5000)
    result.add_argument("--rank", type=int, default=50)
    result.add_argument("--k", type=int, default=100)
    result.add_argument("--seed", type=int, default=17)
    result.add_argument("--sectors", type=int, default=20)
    result.add_argument("--styles", type=int, default=8)
    result.add_argument("--stresses", type=int, default=10)
    result.add_argument("--gamma-scale", type=float, default=100.0)
    result.add_argument("--sector-band", type=float, default=0.08)
    result.add_argument("--style-band", type=float, default=0.20)
    result.add_argument("--stress-band", type=float, default=0.01)
    result.add_argument("--relaxation-backend", default="fista")
    result.add_argument("--relaxation-tolerance", type=float, default=1e-7)
    result.add_argument("--relaxation-iterations", type=int, default=10000)
    result.add_argument("--restricted-solver", default="osqp")
    result.add_argument("--time-limit", type=float, default=600.0)
    result.add_argument("--node-limit", type=int, default=1_000_000)
    result.add_argument("--relative-gap", type=float, default=0.0)
    result.add_argument("--absolute-gap", type=float, default=0.0)
    result.add_argument("--node-dual-iterations", type=int, default=15)
    result.add_argument(
        "--node-heuristic-frequency",
        type=int,
        default=1,
        help="solve a candidate support QP every N processed nodes",
    )
    result.add_argument(
        "--branching-rule",
        choices=("max_min", "product", "max"),
        default="max_min",
    )
    result.add_argument("--no-screening", action="store_true")
    result.add_argument(
        "--cuts",
        action="store_true",
        help="enable the safe multi-selector cut variant",
    )
    result.add_argument(
        "--gurobi-time-limit",
        type=float,
        default=0.0,
        help=(
            "positive values enable the matched perspective-strengthened "
            "Gurobi MIQCP run"
        ),
    )
    result.add_argument(
        "--mosek-time-limit",
        type=float,
        default=0.0,
        help="positive values enable the matched MOSEK MISOCP run",
    )
    result.add_argument("--threads", type=int, default=1)
    return result


def _instance(args: argparse.Namespace):
    if args.bundle is not None:
        return load_instance_bundle(args.bundle)
    return generate_instance(
        dimension=args.dimension,
        rank=args.rank,
        k=args.k,
        gamma_scale=args.gamma_scale,
        regime="hybrid",
        seed=args.seed,
        sectors=args.sectors,
        style_factors=args.styles,
        stress_constraints=args.stresses,
        target_fraction=0.3,
        target_iterations=200,
        sector_band=args.sector_band,
        style_band=args.style_band,
        stress_band=args.stress_band,
        annual_volatility=0.20,
        common_correlation=0.15,
    )


def _bnb_record(result: Any, preprocessing_seconds: float) -> dict[str, Any]:
    raw = result.raw
    return {
        "solver": "sksfolio_bnb",
        "status": result.status,
        "upper_bound": result.upper_bound,
        "lower_bound": result.lower_bound,
        "absolute_gap": result.absolute_gap,
        "relative_gap": result.relative_gap,
        "preprocessing_seconds": preprocessing_seconds,
        "search_seconds": raw.get("solve_seconds"),
        "end_to_end_seconds": preprocessing_seconds
        + float(raw.get("solve_seconds", 0.0)),
        "nodes": raw.get("nodes_processed"),
        "open_nodes": raw.get("open_nodes"),
        "root_screened": raw.get("root_screened_count"),
        "active_cuts": raw.get("cut_statistics", {}).get("active_cuts"),
        "cut_fixings": raw.get("cut_fixings"),
        "restricted_qp_solves": raw.get("restricted_qp_solves"),
        "node_dual_calls": raw.get("node_dual_calls"),
        "node_dual_seconds": raw.get("node_dual_seconds"),
        "restricted_qp_seconds": raw.get("restricted_qp_seconds"),
        "cut_literals_removed": raw.get("cut_literals_removed"),
        "cut_shrink_bound_evaluations": raw.get(
            "cut_shrink_bound_evaluations"
        ),
        "floating_point_certified": raw.get("floating_point_certified"),
    }


def _commercial_record(
    result: Any,
    preprocessing_seconds: float,
) -> dict[str, Any]:
    raw = result.raw
    lower = raw.get("solver_objective_bound")
    upper = result.upper_bound
    total_seconds = float(
        raw.get("total_seconds", raw.get("solve_seconds", 0.0))
    )
    return {
        "solver": f"{raw.get('solver', 'commercial')}_mixed_integer",
        "formulation": raw.get("formulation", "exact_binary_miqp"),
        "status": result.status,
        "upper_bound": upper,
        "lower_bound": lower,
        "absolute_gap": (
            None if upper is None or lower is None else max(0.0, upper - lower)
        ),
        "relative_gap": (
            None
            if upper is None or lower is None
            else max(0.0, upper - lower) / max(1.0, abs(upper))
        ),
        "shared_warm_start_seconds": preprocessing_seconds,
        "model_build_seconds": raw.get("build_seconds"),
        "optimizer_seconds": raw.get(
            "optimizer_seconds",
            raw.get("solve_seconds"),
        ),
        "commercial_total_seconds": total_seconds,
        "end_to_end_seconds": preprocessing_seconds + total_seconds,
        "nodes": raw.get("node_count"),
        "solution_count": raw.get("solution_count"),
        "error": raw.get("error"),
    }


def _comparison(records: list[dict[str, Any]]) -> dict[str, Any]:
    custom = next(
        (row for row in records if row["solver"] == "sksfolio_bnb"),
        None,
    )
    if custom is None:
        return {}
    custom_upper = custom.get("upper_bound")
    custom_time = custom.get("end_to_end_seconds")
    comparisons = []
    for row in records:
        if row is custom:
            continue
        upper = row.get("upper_bound")
        elapsed = row.get("end_to_end_seconds")
        comparisons.append(
            {
                "solver": row["solver"],
                "status": row.get("status"),
                "objective_difference_custom_minus_commercial": (
                    None
                    if custom_upper is None or upper is None
                    else float(custom_upper) - float(upper)
                ),
                "commercial_over_custom_time_ratio": (
                    None
                    if custom_time is None
                    or elapsed is None
                    or float(custom_time) <= 0.0
                    else float(elapsed) / float(custom_time)
                ),
            }
        )
    return {
        "custom_solver": "sksfolio_bnb",
        "objective_agreement_is_meaningful_only_when_both_statuses_are_optimal": True,
        "commercial_comparisons": comparisons,
    }


def main(arguments: Optional[Sequence[str]] = None) -> None:
    args = parser().parse_args(arguments)
    instance = _instance(args)

    preprocessing_start = time.perf_counter()
    relaxation = solve_relaxation(
        instance,
        backend=args.relaxation_backend,
        options={
            "tolerance": args.relaxation_tolerance,
            "max_iterations": args.relaxation_iterations,
            "threads": args.threads,
        },
    )
    incumbent = solve_incumbent(
        instance,
        relaxation,
        method="auto",
        restricted_solver=args.restricted_solver,
        random_state=args.seed,
    )
    preprocessing_seconds = time.perf_counter() - preprocessing_start
    matched_absolute_gap = float(args.absolute_gap)
    if incumbent.upper_bound is not None:
        matched_absolute_gap = max(
            matched_absolute_gap,
            float(args.relative_gap)
            * max(1.0, abs(float(incumbent.upper_bound))),
        )
    matched_relative_gap = 0.0

    bnb = solve_bnb(
        instance,
        relaxation=relaxation,
        incumbent=(incumbent if incumbent.feasible else None),
        restricted_solver=args.restricted_solver,
        time_limit=args.time_limit,
        node_limit=args.node_limit,
        relative_gap=matched_relative_gap,
        absolute_gap=matched_absolute_gap,
        options={
            "safe_screening": not args.no_screening,
            "multi_selector_cuts": args.cuts,
            "root_pair_cuts": args.cuts,
            "node_dual_iterations": args.node_dual_iterations,
            "node_heuristic_frequency": args.node_heuristic_frequency,
            "branching_rule": args.branching_rule,
            "relaxation_options": {"threads": args.threads},
        },
    )
    records = [_bnb_record(bnb, preprocessing_seconds)]

    if args.gurobi_time_limit > 0.0:
        try:
            gurobi = solve_gurobi_incumbent(
                instance,
                warm_start=(incumbent if incumbent.feasible else None),
                time_limit=args.gurobi_time_limit,
                options={
                    "verbose": False,
                    "Threads": args.threads,
                    "MIPFocus": 0,
                    "MIPGap": matched_relative_gap,
                    "MIPGapAbs": matched_absolute_gap,
                    "FeasibilityTol": 1e-9,
                    "OptimalityTol": 1e-9,
                    "BarQCPConvTol": 1e-9,
                    "Seed": args.seed,
                },
            )
            records.append(_commercial_record(gurobi, preprocessing_seconds))
        except ImportError as error:
            records.append(
                {
                    "solver": "gurobi_mixed_integer",
                    "status": "unavailable",
                    "error": str(error),
                }
            )

    if args.mosek_time_limit > 0.0:
        try:
            mosek = solve_mosek_incumbent(
                instance,
                warm_start=(incumbent if incumbent.feasible else None),
                time_limit=args.mosek_time_limit,
                options={
                    "verbose": False,
                    "threads": args.threads,
                    "relative_gap": matched_relative_gap,
                    "absolute_gap": matched_absolute_gap,
                },
            )
            records.append(_commercial_record(mosek, preprocessing_seconds))
        except ImportError as error:
            records.append(
                {
                    "solver": "mosek_mixed_integer",
                    "status": "unavailable",
                    "error": str(error),
                }
            )

    report = {
        "instance": {
            "dimension": instance.dimension,
            "rank": instance.rank,
            "k": instance.k,
            "constraint_rows": instance.rows,
            "constraint_nnz": int(instance.C.nnz),
            "seed": args.seed,
        },
        "root_relaxation": {
            "backend": args.relaxation_backend,
            "status": relaxation.status,
            "safe_dual_bound": relaxation.safe_dual_bound,
            "incumbent_upper_bound": incumbent.upper_bound,
            "preprocessing_seconds": preprocessing_seconds,
        },
        "results": records,
        "comparison": _comparison(records),
        "matched_gap_policy": {
            "requested_relative_gap": args.relative_gap,
            "requested_absolute_gap": args.absolute_gap,
            "effective_relative_gap": matched_relative_gap,
            "effective_absolute_gap": matched_absolute_gap,
            "rule": (
                "one absolute cutoff derived from the shared incumbent, "
                "then passed unchanged to all three solvers"
            ),
        },
        "comparison_note": (
            "All searches receive the same feasible warm start; sksfolio "
            "also uses the root relaxation certificate produced during the "
            "reported preprocessing phase. Gurobi uses the exact binary "
            "perspective MIQCP and MOSEK uses its equivalent exact MISOCP. "
            "Timings are wall-clock values."
        ),
    }
    payload = json.dumps(report, indent=2)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
