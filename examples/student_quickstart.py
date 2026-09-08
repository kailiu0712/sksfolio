"""End-to-end example for the two corrected relaxation algorithms."""

from __future__ import annotations

import argparse

from sksfolio import solve_bnb, solve_incumbent, solve_relaxation
from sksfolio.benchmarks.instance_generator import generate_instance


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dimension", type=int, default=80)
    result.add_argument("--rank", type=int, default=10)
    result.add_argument("--k", type=int, default=8)
    result.add_argument("--seed", type=int, default=17)
    result.add_argument("--threads", type=int, default=1)
    result.add_argument("--solver-time-limit", type=float, default=10.0)
    result.add_argument("--bnb-time-limit", type=float, default=20.0)
    result.add_argument("--skip-bnb", action="store_true")
    return result


def make_problem(args: argparse.Namespace):
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


def main() -> None:
    args = parser().parse_args()
    problem = make_problem(args)
    common = {
        "tolerance": 1e-6,
        "max_iterations": 5_000,
        "time_limit": args.solver_time_limit,
        "threads": args.threads,
    }
    relaxations = {}
    for algorithm in ("corrected_fista", "corrected_lbfgs"):
        result = solve_relaxation(problem, backend=algorithm, options=common)
        relaxations[algorithm] = result
        print(
            f"{algorithm:24s} status={result.status:18s} "
            f"objective={result.objective!s:>14s} "
            f"safe_dual={result.safe_dual_bound!s:>14s} "
            f"seconds={result.raw.get('end_to_end_seconds')!s}"
        )

    root = relaxations["corrected_lbfgs"]
    incumbent = solve_incumbent(
        problem,
        root,
        method="auto",
        restricted_solver="osqp",
        random_state=args.seed,
    )
    print(
        f"osqp_incumbent           status={incumbent.status:18s} "
        f"upper_bound={incumbent.upper_bound!s:>14s}"
    )

    if not args.skip_bnb:
        result = solve_bnb(
            problem,
            relaxation=root,
            incumbent=incumbent if incumbent.feasible else None,
            restricted_solver="osqp",
            time_limit=args.bnb_time_limit,
            relative_gap=1e-4,
            absolute_gap=1e-8,
        )
        print(
            f"safe_screened_bnb       status={result.status:18s} "
            f"upper={result.upper_bound!s:>14s} "
            f"lower={result.lower_bound!s:>14s} "
            f"nodes={result.nodes_processed}"
        )


if __name__ == "__main__":
    main()
