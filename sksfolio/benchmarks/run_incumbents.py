"""Compare sparse incumbent heuristics on constrained Markowitz instances."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional, Sequence

from ..incumbent import solve_gurobi_incumbent, solve_incumbent
from ..relaxation import load_instance_bundle, solve_relaxation
from .instance_generator import generate_instance


DEFAULT_METHODS = (
    "topk",
    "binary_prox",
    "randomized",
    "prune",
    "discrete_first_order",
    "swap",
    "auto",
    "quality",
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Benchmark feasible sparse portfolio incumbents"
    )
    result.add_argument("--bundle", type=Path, default=None)
    result.add_argument("--output", type=Path, default=None)
    result.add_argument("--dimension", type=int, default=5000)
    result.add_argument("--rank", type=int, default=50)
    result.add_argument("--k", type=int, default=50)
    result.add_argument("--seed", type=int, default=7)
    result.add_argument("--sectors", type=int, default=20)
    result.add_argument("--styles", type=int, default=8)
    result.add_argument("--stresses", type=int, default=10)
    result.add_argument("--gamma-scale", type=float, default=100.0)
    result.add_argument("--relaxation-backend", default="fista")
    result.add_argument("--relaxation-tolerance", type=float, default=1e-5)
    result.add_argument("--relaxation-iterations", type=int, default=5000)
    result.add_argument("--restricted-solver", default="auto")
    result.add_argument("--random-samples", type=int, default=16)
    result.add_argument("--heuristic-time-limit", type=float, default=None)
    result.add_argument("--gurobi-time-limit", type=float, default=0.0)
    result.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
    )
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
        sector_band=0.025,
        style_band=0.10,
        stress_band=0.005,
        annual_volatility=0.20,
        common_correlation=0.15,
    )


def main(arguments: Optional[Sequence[str]] = None) -> None:
    args = parser().parse_args(arguments)
    instance = _instance(args)
    relaxation = solve_relaxation(
        instance,
        backend=args.relaxation_backend,
        options={
            "tolerance": args.relaxation_tolerance,
            "max_iterations": args.relaxation_iterations,
            "threads": 1,
        },
    )
    methods = tuple(
        value.strip() for value in args.methods.split(",") if value.strip()
    )
    records = []
    best = None
    for method in methods:
        result = solve_incumbent(
            instance,
            relaxation,
            method=method,
            restricted_solver=args.restricted_solver,
            time_limit=args.heuristic_time_limit,
            random_state=args.seed,
            options={
                "random_samples": args.random_samples,
                "dfo_iterations": 8,
                "maximum_swap_evaluations": 32,
                "maximum_prune_refits": 80,
            },
        )
        if result.feasible and (
            best is None or result.upper_bound < best.upper_bound
        ):
            best = result
        lower = result.raw.get("relaxation_lower_bound")
        records.append(
            {
                "solver": method,
                "status": result.status,
                "upper_bound": result.upper_bound,
                "relaxation_lower_bound": lower,
                "absolute_gap": (
                    result.upper_bound - lower
                    if result.upper_bound is not None and lower is not None
                    else None
                ),
                "solve_seconds": result.raw.get("solve_seconds"),
                "restricted_qp_seconds": result.raw.get(
                    "restricted_qp_seconds"
                ),
                "candidates": result.raw.get("candidate_count"),
                "feasible_candidates": result.raw.get(
                    "feasible_candidate_count"
                ),
                "winning_method": result.raw.get("winning_method"),
                "cardinality": (
                    int(result.active_support.size) if result.feasible else None
                ),
                "maximum_violation": (
                    result.raw["diagnostics"]["violations"]["maximum"]
                    if result.feasible
                    else None
                ),
            }
        )

    if args.gurobi_time_limit > 0.0:
        gurobi = solve_gurobi_incumbent(
            instance,
            warm_start=best,
            time_limit=args.gurobi_time_limit,
            options={"verbose": False},
        )
        records.append(
            {
                "solver": "gurobi_miqp",
                "status": gurobi.status,
                "upper_bound": gurobi.upper_bound,
                "relaxation_lower_bound": gurobi.raw.get(
                    "solver_objective_bound"
                ),
                "absolute_gap": (
                    gurobi.upper_bound
                    - gurobi.raw["solver_objective_bound"]
                    if gurobi.upper_bound is not None
                    and gurobi.raw.get("solver_objective_bound") is not None
                    else None
                ),
                "solve_seconds": gurobi.raw.get("solve_seconds"),
                "restricted_qp_seconds": None,
                "candidates": gurobi.raw.get("node_count"),
                "feasible_candidates": gurobi.raw.get("solution_count"),
                "winning_method": None,
                "cardinality": (
                    int(gurobi.active_support.size) if gurobi.feasible else None
                ),
                "maximum_violation": (
                    gurobi.raw["diagnostics"]["violations"]["maximum"]
                    if gurobi.feasible
                    else None
                ),
            }
        )

    report = {
        "instance": {
            "dimension": instance.dimension,
            "rank": instance.rank,
            "k": instance.k,
            "constraint_rows": instance.rows,
        },
        "relaxation": {
            "backend": args.relaxation_backend,
            "status": relaxation.status,
            "objective": relaxation.objective,
            "safe_dual_bound": relaxation.safe_dual_bound,
            "solve_seconds": relaxation.raw.get("end_to_end_seconds"),
        },
        "results": records,
    }
    print(json.dumps(report, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)


if __name__ == "__main__":
    main()
