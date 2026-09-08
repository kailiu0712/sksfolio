"""Run the retained corrected-relaxation, incumbent, screening, and BnB flow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from ..bnb import solve_bnb
from ..incumbent import solve_incumbent
from ..relaxation import (
    CORRECTED_ALGORITHMS,
    load_instance_bundle,
    solve_relaxation,
)
from .instance_generator import generate_instance


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--bundle", type=Path, default=None)
    result.add_argument("--output", type=Path, default=None)
    result.add_argument("--dimension", type=int, default=100)
    result.add_argument("--rank", type=int, default=10)
    result.add_argument("--k", type=int, default=10)
    result.add_argument("--seed", type=int, default=17)
    result.add_argument("--threads", type=int, default=1)
    result.add_argument(
        "--relaxation-backend",
        choices=CORRECTED_ALGORITHMS,
        default="corrected_lbfgs",
    )
    result.add_argument("--relaxation-tolerance", type=float, default=1e-6)
    result.add_argument("--relaxation-iterations", type=int, default=5_000)
    result.add_argument("--time-limit", type=float, default=60.0)
    result.add_argument("--node-limit", type=int, default=100_000)
    result.add_argument("--relative-gap", type=float, default=1e-4)
    result.add_argument("--absolute-gap", type=float, default=1e-8)
    result.add_argument("--no-screening", action="store_true")
    return result


def _instance(args: argparse.Namespace):
    if args.bundle is not None:
        return load_instance_bundle(args.bundle)
    return generate_instance(
        dimension=args.dimension,
        rank=args.rank,
        k=args.k,
        gamma_scale=100.0,
        regime="hybrid",
        seed=args.seed,
        sectors=min(5, args.k, args.dimension),
        style_factors=3,
        stress_constraints=5,
        target_fraction=0.3,
        target_iterations=200,
        sector_band=0.08,
        style_band=0.20,
        stress_band=0.01,
        annual_volatility=0.20,
        common_correlation=0.15,
    )


def main(arguments: Optional[Sequence[str]] = None) -> None:
    args = parser().parse_args(arguments)
    problem = _instance(args)
    relaxation = solve_relaxation(
        problem,
        args.relaxation_backend,
        options={
            "threads": args.threads,
            "tolerance": args.relaxation_tolerance,
            "max_iterations": args.relaxation_iterations,
        },
    )
    incumbent = solve_incumbent(
        problem,
        relaxation,
        method="auto",
        restricted_solver="osqp",
        random_state=args.seed,
    )
    result = solve_bnb(
        problem,
        relaxation=relaxation,
        incumbent=incumbent if incumbent.feasible else None,
        restricted_solver="osqp",
        time_limit=args.time_limit,
        node_limit=args.node_limit,
        relative_gap=args.relative_gap,
        absolute_gap=args.absolute_gap,
        options={"safe_screening": not args.no_screening},
    )
    report = {
        "instance": {
            "dimension": problem.dimension,
            "rank": problem.rank,
            "k": problem.k,
            "constraint_rows": problem.rows,
            "seed": args.seed,
        },
        "relaxation": {
            "algorithm": args.relaxation_backend,
            "status": relaxation.status,
            "safe_dual_bound": relaxation.safe_dual_bound,
        },
        "incumbent": {
            "status": incumbent.status,
            "upper_bound": incumbent.upper_bound,
        },
        "bnb": {
            "status": result.status,
            "upper_bound": result.upper_bound,
            "lower_bound": result.lower_bound,
            "absolute_gap": result.absolute_gap,
            "relative_gap": result.relative_gap,
            "nodes": result.nodes_processed,
            "seconds": result.raw.get("solve_seconds"),
        },
    }
    payload = json.dumps(report, indent=2)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
