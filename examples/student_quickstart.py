"""Small end-to-end example for the public sksfolio APIs.

The default run uses only NumPy, SciPy, and OSQP. Commercial relaxation
backends and JuMP are opt-in because they require separate installations.
"""

from __future__ import annotations

import argparse
from typing import Any

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
    result.add_argument(
        "--commercial",
        action="store_true",
        help="also run the native Python Gurobi and MOSEK relaxations",
    )
    result.add_argument(
        "--jump-optimizer",
        default=None,
        help="also run JuMP with this optimizer, for example clarabel",
    )
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


def show_relaxation(label: str, result: Any) -> None:
    print(
        f"{label:24s} status={result.status:18s} "
        f"objective={result.objective!s:>14s} "
        f"safe_dual={result.safe_dual_bound!s:>14s} "
        f"seconds={result.raw.get('end_to_end_seconds')!s}"
    )


def main() -> None:
    args = parser().parse_args()
    problem = make_problem(args)
    names = tuple(problem.constraint_names)
    print(
        "constraint stack: "
        f"rows={problem.rows}, "
        f"budget={sum(name == 'budget' for name in names)}, "
        f"minimum_return={sum(name == 'minimum_return' for name in names)}, "
        f"sectors={sum(name.startswith('sector_') for name in names)}, "
        f"styles={sum(name.startswith('style_') for name in names)}, "
        f"one_sided_stresses="
        f"{sum(name.startswith('stress_') for name in names)}"
    )
    common = {
        "tolerance": 1e-6,
        "max_iterations": 5_000,
        "time_limit": args.solver_time_limit,
        "threads": args.threads,
    }
    methods = (
        (
            "fista_dual_fista",
            "fista",
            {
                **common,
                "prox_oracle": "dual_fista",
                "restart_strategy": "gradient",
            },
        ),
        (
            "fista_dual_lbfgs",
            "fista",
            {
                **common,
                "prox_oracle": "dual_lbfgs",
                "restart_strategy": "gradient",
            },
        ),
    )

    relaxations = {}
    for label, backend, options in methods:
        result = solve_relaxation(
            problem,
            backend=backend,
            pava="partial_sort",
            options=options,
        )
        relaxations[label] = result
        show_relaxation(label, result)

    if args.commercial:
        for backend in ("gurobi", "mosek"):
            result = solve_relaxation(
                problem,
                backend=backend,
                options={
                    "tolerance": 1e-6,
                    "time_limit": args.solver_time_limit,
                    "threads": args.threads,
                    "log": False,
                },
            )
            show_relaxation(f"{backend}_python", result)

    if args.jump_optimizer:
        result = solve_relaxation(
            problem,
            backend="jump",
            options={
                "optimizer": args.jump_optimizer,
                "julia_instantiate": True,
                "tolerance": 1e-6,
                "time_limit": args.solver_time_limit,
                "threads": args.threads,
            },
        )
        show_relaxation(f"jump_{args.jump_optimizer}", result)

    root = relaxations["fista_dual_lbfgs"]
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
            options={
                "safe_screening": True,
                "multi_selector_cuts": False,
            },
        )
        print(
            f"safe_screened_bnb       status={result.status:18s} "
            f"upper={result.upper_bound!s:>14s} "
            f"lower={result.lower_bound!s:>14s} "
            f"nodes={result.nodes_processed}"
        )


if __name__ == "__main__":
    main()
