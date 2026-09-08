"""Benchmark the two supported corrected relaxation algorithms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from ..relaxation import CORRECTED_ALGORITHMS, solve_relaxation
from .instance_generator import generate_instance


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dimension", type=int, default=100)
    result.add_argument("--rank", type=int, default=10)
    result.add_argument("--k", type=int, default=10)
    result.add_argument("--seed", type=int, default=17)
    result.add_argument("--threads", type=int, default=1)
    result.add_argument("--time-limit", type=float, default=30.0)
    result.add_argument("--tolerance", type=float, default=1e-6)
    result.add_argument("--max-iterations", type=int, default=5_000)
    result.add_argument("--output", type=Path, default=None)
    return result


def main(arguments: Optional[Sequence[str]] = None) -> None:
    args = parser().parse_args(arguments)
    problem = generate_instance(
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
    records = []
    for algorithm in CORRECTED_ALGORITHMS:
        result = solve_relaxation(
            problem,
            algorithm,
            options={
                "threads": args.threads,
                "time_limit": args.time_limit,
                "tolerance": args.tolerance,
                "max_iterations": args.max_iterations,
            },
        )
        records.append(
            {
                "algorithm": algorithm,
                "status": result.status,
                "objective": result.objective,
                "safe_dual_bound": result.safe_dual_bound,
                "maximum_violation": result.raw.get("violation"),
                "iterations": result.raw.get("iterations"),
                "prox_oracle": result.raw.get("prox_oracle_used"),
                "seconds": result.raw.get("end_to_end_seconds"),
            }
        )
    report = {
        "instance": {
            "dimension": problem.dimension,
            "rank": problem.rank,
            "k": problem.k,
            "constraint_rows": problem.rows,
            "seed": args.seed,
        },
        "results": records,
    }
    payload = json.dumps(report, indent=2)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
