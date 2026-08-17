"""Real-data counterpart to :mod:`run_four_scaling`.

Same six-panel scaling sweep -- dimension ladder, small-k/large-k rules,
rank rule, constraint-band profile, ``gamma_scale``, ``regime``, solvers,
and CSV/plot schema -- reused unchanged from :mod:`run_four_scaling` by
direct import. The only two things that differ:

1. Data source: instances are built from real daily returns via
   ``historical_factor_data`` (through ``generate_instance``'s
   ``daily_returns`` argument) instead of the synthetic factor model.
2. Stock pool per dimension, sampled from three nested real universes
   prepared by :mod:`wilshire5000_data` (run that module first):

   - ``dimension < 500``   -> sampled from the S&P 500 tier
   - ``500 <= dimension < 1000`` -> sampled from the Russell-1000-proxy tier
   - ``dimension >= 1000`` -> sampled from the Wilshire-5000-proxy tier

   Each nests the smaller tiers, matching sp500 subset russell1000 subset
   wilshire5000 in the real market.

The meaning of "seed" changes accordingly: in the synthetic runner each of
the 10 seeds per (scenario, dimension) drives an independent synthetic
factor draw; here each seed instead draws an independent random n-stock
subset (without replacement) from the applicable pool, so the 10 rows
report the spread across different stock picks rather than different
synthetic draws. That mitigates any single draw's idiosyncratic
stock-selection bias, which is the real-data analogue of averaging over
random seeds in the synthetic case. Dimensions larger than the Wilshire-tier
pool size cannot be sampled and are skipped with a printed warning, since
(unlike synthetic data) a real pool has a hard size limit.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from tqdm.auto import tqdm

from .instance_generator import generate_instance
from .run_four_scaling import (
    BCW_STANDARD_PROFILE,
    EXPERIMENTS,
    METHODS,
    SCENARIOS,
    _commercial_solver,
    _default_result_path,
    _aggregate_rows,
    _bootstrap_commercial_environment,
    _constraint_counts,
    _existing_rows,
    _fista_relaxation_options,
    _plot,
    _prox_instance,
    _prox_problem_data,
    _rank_for_dimension,
    _repository_root,
    _row_base,
    _seed_values,
    _solve_bnb_run,
    _solve_prox,
    _solve_relaxation_run,
    _workspace_root,
    _write_rows,
    _write_summary_rows,
    parser as _synthetic_parser,
)
from ..relaxation import evaluate_solution, perspective_value, solve_relaxation
from ..relaxation.fista import LinearConstraintProx
from .wilshire5000_data import DEFAULT_OUTPUT_DIR as DEFAULT_DATASET_DIR
from .wilshire5000_data import RealDataset, TIER_NAMES, load_dataset


DEFAULT_PLOT = _default_result_path("four_scaling_six_panel_real.svg")
DEFAULT_RESULTS = _default_result_path("four_scaling_results_real.csv")
DEFAULT_ACCURACY_OUTPUT = _default_result_path("four_scaling_accuracy_real.csv")
DEFAULT_ACCURACY_SUMMARY = _default_result_path(
    "four_scaling_accuracy_real_summary.csv"
)


def _pool_tier(dimension: int) -> str:
    if dimension < 500:
        return "sp500"
    if dimension < 1000:
        return "russell1000"
    return "wilshire5000"


def _sample_tickers(pool: Sequence[str], dimension: int, seed: int) -> List[str]:
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(pool), size=dimension, replace=False)
    return [pool[int(index)] for index in indices]


def _available_dimensions(dimensions: Sequence[int], dataset: RealDataset) -> List[int]:
    max_pool = len(dataset.tiers["wilshire5000"])
    kept: List[int] = []
    for dimension in dimensions:
        if dimension > max_pool:
            print(
                f"  [skip] dimension {dimension} exceeds the real dataset's "
                f"largest pool ({max_pool} tickers); skipping"
            )
            continue
        kept.append(int(dimension))
    return kept


def _build_real_instance(
    dimension: int,
    k: int,
    seed: int,
    rank_cap: int,
    dataset: RealDataset,
) -> Any:
    tier = _pool_tier(dimension)
    pool = dataset.tiers[tier]
    if dimension > len(pool):
        raise ValueError(f"{tier} pool has only {len(pool)} tickers, need {dimension}")
    tickers = _sample_tickers(pool, dimension, seed)
    daily_returns = dataset.returns[tickers].to_numpy(dtype=np.float64)
    rank = _rank_for_dimension(dimension, rank_cap)
    sectors, styles, stresses = _constraint_counts(dimension)
    return generate_instance(
        dimension=dimension,
        rank=rank,
        k=k,
        gamma_scale=100.0,
        regime="unconstrained",
        seed=seed,
        sectors=sectors,
        style_factors=styles,
        stress_constraints=stresses,
        target_fraction=0.3,
        target_iterations=200,
        sector_band=float(BCW_STANDARD_PROFILE["sector_band"]),
        style_band=float(BCW_STANDARD_PROFILE["style_band"]),
        stress_band=float(BCW_STANDARD_PROFILE["stress_band"]),
        annual_volatility=0.20,
        common_correlation=0.15,
        daily_returns=daily_returns,
        outlier_mode="cell",
    )


def _write_dict_rows(path: Path, rows: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as stream:
            stream.write("")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _relative_error(value: Optional[float], reference: Optional[float]) -> Optional[float]:
    if value is None or reference is None:
        return None
    return abs(float(value) - float(reference)) / max(1.0, abs(float(reference)))


def _prox_objective(
    instance: Any,
    argument: np.ndarray,
    gamma: float,
    x: np.ndarray,
) -> Optional[float]:
    perspective = perspective_value(x, instance.k, tolerance=1e-8)
    if not math.isfinite(perspective):
        return None
    displacement = np.asarray(x, dtype=float).reshape(-1) - np.asarray(
        argument,
        dtype=float,
    ).reshape(-1)
    return 0.5 * float(displacement @ displacement) + float(gamma) * float(perspective)


def _maximum_violation(diagnostics: Dict[str, Any]) -> Optional[float]:
    violations = diagnostics.get("violations", {})
    value = violations.get("maximum")
    if value is None:
        return None
    return float(value)


def _solve_prox_with_details(
    instance: Any,
    method: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    argument, gamma = _prox_problem_data(instance)
    if method in {"fista_dual_fista", "fista_dual_lbfgs"}:
        dual_solver = "fista" if method.endswith("dual_fista") else "lbfgs"
        oracle = LinearConstraintProx(
            instance.C,
            instance.lower,
            instance.upper,
            instance.k,
            pava_method="partial_sort",
            tolerance=args.prox_tolerance,
            max_iterations=args.prox_max_iterations,
            adaptive_restart=True,
            use_budget_fast_path=False,
            dual_solver=dual_solver,
            lbfgs_memory=args.prox_lbfgs_memory,
            lbfgs_max_line_search=args.prox_lbfgs_max_line_search,
            lbfgs_fallback=args.prox_lbfgs_fallback,
        )
        result = oracle.solve(argument, gamma)
        diagnostics = evaluate_solution(instance, result.x, domain_tolerance=1e-8)
        return {
            "status": "converged" if result.converged else "iteration_limit",
            "weights": np.asarray(result.x, dtype=float).reshape(-1),
            "objective": _prox_objective(instance, argument, gamma, result.x),
            "diagnostics": diagnostics,
            "constraint_violation": float(result.constraint_violation),
            "fixed_point_residual": float(result.fixed_point_residual),
            "notes": result.method,
        }

    backend = "gurobi.python" if method == "gurobi" else "mosek.python"
    options: dict[str, Any] = {
        "threads": args.threads,
        "tolerance": args.prox_tolerance,
        "time_limit": args.prox_time_limit,
        "log": False,
        "warm_start": False,
    }
    raw = _commercial_solver(backend)(_prox_instance(instance, argument, gamma), options=options)
    weights = raw.get("x")
    diagnostics = (
        evaluate_solution(instance, weights, domain_tolerance=1e-8)
        if weights is not None
        else {}
    )
    objective = raw.get("objective")
    if objective is None and weights is not None:
        objective = _prox_objective(
            instance,
            argument,
            gamma,
            np.asarray(weights, dtype=float).reshape(-1),
        )
    return {
        "status": str(raw.get("status", "unknown")),
        "weights": (
            None
            if weights is None
            else np.asarray(weights, dtype=float).reshape(-1)
        ),
        "objective": (
            None
            if objective is None
            else float(objective)
        ),
        "diagnostics": diagnostics,
        "constraint_violation": _maximum_violation(diagnostics),
        "fixed_point_residual": None,
        "notes": raw.get("message", raw.get("error")),
    }


def _solve_relaxation_with_details(
    instance: Any,
    method: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if method == "gurobi":
        backend = "gurobi"
        options = {
            "threads": args.threads,
            "time_limit": args.relaxation_time_limit,
            "tolerance": args.relaxation_tolerance,
            "log": False,
            "warm_start": False,
        }
    else:
        backend = "fista"
        options = _fista_relaxation_options(method, args)
    result = solve_relaxation(
        instance,
        backend=backend,
        warm_start=False,
        options=options,
    )
    diagnostics = result.raw.get("diagnostics", {})
    return {
        "status": result.status,
        "weights": result.weights,
        "objective": result.objective,
        "diagnostics": diagnostics,
        "constraint_violation": _maximum_violation(diagnostics),
        "fixed_point_residual": result.raw.get("residual"),
        "notes": result.raw.get("prox_oracle_used", result.raw.get("error")),
        "primal_feasible": bool(result.primal_feasible),
    }


def _accuracy_row(
    *,
    scenario: str,
    experiment: str,
    method: str,
    dimension: int,
    k: int,
    rank: int,
    seed: int,
    seed_index: int,
    oracle: Dict[str, Any],
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    oracle_x = oracle.get("weights")
    candidate_x = candidate.get("weights")
    linf_error = None
    l2_error = None
    if oracle_x is not None and candidate_x is not None:
        difference = np.asarray(candidate_x, dtype=float) - np.asarray(
            oracle_x,
            dtype=float,
        )
        linf_error = float(np.max(np.abs(difference)))
        l2_error = float(np.linalg.norm(difference))
    oracle_objective = oracle.get("objective")
    candidate_objective = candidate.get("objective")
    objective_gap = (
        None
        if oracle_objective is None or candidate_objective is None
        else float(candidate_objective) - float(oracle_objective)
    )
    return {
        "scenario": scenario,
        "experiment": experiment,
        "method": method,
        "dimension": int(dimension),
        "k": int(k),
        "rank": int(rank),
        "seed": int(seed),
        "seed_index": int(seed_index),
        "oracle_method": "gurobi",
        "oracle_status": oracle.get("status"),
        "candidate_status": candidate.get("status"),
        "oracle_objective": oracle_objective,
        "candidate_objective": candidate_objective,
        "objective_gap_to_oracle": objective_gap,
        "objective_relative_error": _relative_error(
            candidate_objective,
            oracle_objective,
        ),
        "linf_weight_error": linf_error,
        "l2_weight_error": l2_error,
        "oracle_constraint_violation": oracle.get("constraint_violation"),
        "candidate_constraint_violation": candidate.get("constraint_violation"),
        "candidate_fixed_point_residual": candidate.get("fixed_point_residual"),
        "oracle_notes": oracle.get("notes"),
        "candidate_notes": candidate.get("notes"),
        "oracle_has_solution": oracle_x is not None,
        "candidate_has_solution": candidate_x is not None,
        "comparison_valid": bool(oracle_x is not None and candidate_x is not None),
    }


def _summarize_accuracy(rows: Sequence[Dict[str, Any]]) -> list[Dict[str, Any]]:
    grouped: Dict[tuple[str, str, str], list[Dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["scenario"]), str(row["experiment"]), str(row["method"]))
        grouped.setdefault(key, []).append(row)

    summary: list[Dict[str, Any]] = []
    for (scenario, experiment, method), group in sorted(grouped.items()):
        valid = [row for row in group if row.get("comparison_valid")]
        relative_errors = [
            float(row["objective_relative_error"])
            for row in valid
            if row.get("objective_relative_error") is not None
        ]
        linf_errors = [
            float(row["linf_weight_error"])
            for row in valid
            if row.get("linf_weight_error") is not None
        ]
        candidate_violations = [
            float(row["candidate_constraint_violation"])
            for row in valid
            if row.get("candidate_constraint_violation") is not None
        ]
        worst_row = None
        if relative_errors:
            worst_row = max(
                valid,
                key=lambda row: float(row.get("objective_relative_error", -math.inf)),
            )
        summary.append(
            {
                "scenario": scenario,
                "experiment": experiment,
                "method": method,
                "cases": len(group),
                "valid_cases": len(valid),
                "candidate_successes": sum(
                    str(row.get("candidate_status", "")).lower()
                    in {"optimal", "converged"}
                    for row in group
                ),
                "mean_objective_relative_error": (
                    float(np.mean(relative_errors)) if relative_errors else None
                ),
                "max_objective_relative_error": (
                    float(np.max(relative_errors)) if relative_errors else None
                ),
                "mean_linf_weight_error": (
                    float(np.mean(linf_errors)) if linf_errors else None
                ),
                "max_linf_weight_error": (
                    float(np.max(linf_errors)) if linf_errors else None
                ),
                "max_candidate_constraint_violation": (
                    float(np.max(candidate_violations))
                    if candidate_violations
                    else None
                ),
                "count_relerr_le_1e_8": sum(error <= 1e-8 for error in relative_errors),
                "count_relerr_le_1e_6": sum(error <= 1e-6 for error in relative_errors),
                "count_relerr_le_1e_4": sum(error <= 1e-4 for error in relative_errors),
                "worst_dimension": (
                    None if worst_row is None else int(worst_row["dimension"])
                ),
                "worst_seed_index": (
                    None if worst_row is None else int(worst_row["seed_index"])
                ),
                "worst_objective_relative_error": (
                    None
                    if worst_row is None
                    else worst_row.get("objective_relative_error")
                ),
                "worst_linf_weight_error": (
                    None if worst_row is None else worst_row.get("linf_weight_error")
                ),
            }
        )
    return summary


def _run_accuracy_verification(
    args: argparse.Namespace,
    dataset: RealDataset,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    print("Accuracy verification against Gurobi (real data)")
    rows: list[Dict[str, Any]] = []
    progress_total = (
        len(SCENARIOS)
        * len(args.accuracy_experiments)
        * len(args.accuracy_dimensions)
        * args.accuracy_seed_count
        * 2
    )
    progress = tqdm(total=progress_total, desc="accuracy-real", unit="solve")
    for scenario, k_rule in SCENARIOS:
        for experiment in args.accuracy_experiments:
            for dimension in args.accuracy_dimensions:
                k = k_rule(int(dimension))
                seeds = _seed_values(
                    args.seed,
                    scenario,
                    int(dimension),
                    args.accuracy_seed_count,
                )
                for seed_index, seed in enumerate(seeds):
                    print(
                        f"  verify scenario={scenario} experiment={experiment} "
                        f"n={dimension} k={k} seed={seed} seed_index={seed_index}"
                    )
                    instance = _build_real_instance(
                        int(dimension),
                        int(k),
                        int(seed),
                        int(args.rank_cap),
                        dataset,
                    )
                    if experiment == "prox":
                        oracle = _solve_prox_with_details(instance, "gurobi", args)
                    else:
                        oracle = _solve_relaxation_with_details(instance, "gurobi", args)
                    for method in ("fista_dual_fista", "fista_dual_lbfgs"):
                        if experiment == "prox":
                            candidate = _solve_prox_with_details(instance, method, args)
                        else:
                            candidate = _solve_relaxation_with_details(
                                instance,
                                method,
                                args,
                            )
                        rows.append(
                            _accuracy_row(
                                scenario=scenario,
                                experiment=experiment,
                                method=method,
                                dimension=int(dimension),
                                k=int(k),
                                rank=int(instance.rank),
                                seed=int(seed),
                                seed_index=int(seed_index),
                                oracle=oracle,
                                candidate=candidate,
                            )
                        )
                        progress.update(1)
    progress.close()
    summary = _summarize_accuracy(rows)
    return rows, summary


def _print_accuracy_summary(summary_rows: Sequence[Dict[str, Any]]) -> None:
    print("Accuracy summary vs Gurobi oracle")
    for row in summary_rows:
        print(
            "  "
            f"{row['scenario']} | {row['experiment']} | {row['method']} | "
            f"valid={row['valid_cases']}/{row['cases']} | "
            f"max_relerr={row['max_objective_relative_error']} | "
            f"mean_relerr={row['mean_objective_relative_error']} | "
            f"max_linf={row['max_linf_weight_error']} | "
            f"max_violation={row['max_candidate_constraint_violation']}"
        )


def parser() -> argparse.ArgumentParser:
    result = _synthetic_parser()
    result.description = (
        "Run the same six scaling plots as run_four_scaling.py, but built "
        "from real daily returns sampled from nested S&P 500 / "
        "Russell-1000-proxy / Wilshire-5000-proxy stock pools instead of "
        "the synthetic factor model"
    )
    result.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="directory produced by wilshire5000_data.py",
    )
    result.add_argument(
        "--verify-accuracy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="compare real-data prox/relaxation solutions against Gurobi",
    )
    result.add_argument(
        "--accuracy-output",
        type=Path,
        default=DEFAULT_ACCURACY_OUTPUT,
    )
    result.add_argument(
        "--accuracy-summary-output",
        type=Path,
        default=DEFAULT_ACCURACY_SUMMARY,
    )
    result.add_argument(
        "--accuracy-experiments",
        nargs="+",
        choices=("prox", "relaxation"),
        default=("prox", "relaxation"),
    )
    result.add_argument(
        "--accuracy-dimensions",
        nargs="+",
        type=int,
        default=(10, 30, 100, 300),
    )
    result.add_argument(
        "--accuracy-seed-count",
        type=int,
        default=3,
    )
    result.set_defaults(output=DEFAULT_RESULTS, plot=DEFAULT_PLOT)
    return result


def main(arguments: Optional[Sequence[str]] = None) -> None:
    args = parser().parse_args(arguments)
    env_info = _bootstrap_commercial_environment(args.gurobi_license_mode)

    print("Four-scaling benchmark (real data)")
    print(f"Workspace root: {_workspace_root()}")
    print(f"Repository root: {_repository_root()}")
    print(f"Gurobi license: {env_info.get('gurobi_license')}")
    print(f"MOSEK license: {env_info.get('mosek_license')}")
    print(f"MOSEK site-packages: {env_info.get('mosek_site_packages')}")
    print(f"Dataset directory: {args.dataset_dir}")

    dataset = load_dataset(args.dataset_dir)
    for tier in TIER_NAMES:
        print(f"  pool {tier}: {len(dataset.tiers[tier])} tickers")

    dimensions = _available_dimensions(args.dimensions, dataset)
    print(f"Dimensions: {tuple(dimensions)}")
    print(f"Seeds per dimension: {args.seed_count}")

    existing = _existing_rows(args.output) if args.resume else {}
    keys_to_run: list[tuple[str, str, int, str, int]] = []
    for scenario, _ in SCENARIOS:
        for experiment in EXPERIMENTS:
            for dimension in dimensions:
                seeds = _seed_values(args.seed, scenario, int(dimension), args.seed_count)
                for seed_index, _ in enumerate(seeds):
                    for method in METHODS:
                        key = (scenario, experiment, int(dimension), method, int(seed_index))
                        if key not in existing:
                            keys_to_run.append(key)

    print(f"Cached rows reused: {len(existing)}")
    print(f"Rows to run now: {len(keys_to_run)}")
    progress = tqdm(total=len(keys_to_run), desc="four-scaling-real", unit="run")
    fresh_rows: list[dict[str, Any]] = []

    for scenario, k_rule in SCENARIOS:
        print(f"[scenario] {scenario}")
        for experiment in EXPERIMENTS:
            print(f"  [experiment] {experiment}")
            experiment_bar = tqdm(dimensions, desc=f"{scenario}:{experiment}", leave=False, unit="n")
            for dimension in experiment_bar:
                k = k_rule(int(dimension))
                seeds = _seed_values(args.seed, scenario, int(dimension), args.seed_count)
                experiment_bar.set_postfix({"n": dimension, "k": k})
                seed_bar = tqdm(
                    list(enumerate(seeds)),
                    desc=f"{scenario}:{experiment}:n{dimension}",
                    leave=False,
                    unit="seed",
                )
                for seed_index, seed in seed_bar:
                    try:
                        instance = _build_real_instance(
                            int(dimension), k, int(seed), int(args.rank_cap), dataset
                        )
                        build_error = None
                    except Exception as error:
                        instance = None
                        build_error = f"{type(error).__name__}: {error}"
                    seed_bar.set_postfix({"seed": seed_index + 1, "of": len(seeds)})

                    for method in METHODS:
                        key = (scenario, experiment, int(dimension), method, int(seed_index))
                        if key in existing:
                            continue
                        print(
                            f"    running scenario={scenario} experiment={experiment} "
                            f"n={dimension} k={k} seed={seed} "
                            f"seed_index={seed_index} method={method}"
                        )
                        base = _row_base(
                            scenario,
                            experiment,
                            method,
                            int(dimension),
                            int(k),
                            int(instance.rank) if instance is not None else -1,
                            int(seed),
                            int(seed_index),
                        )
                        if instance is None:
                            details = {
                                "status": "error",
                                "elapsed_seconds": math.nan,
                                "solver_seconds": math.nan,
                                "objective": None,
                                "iterations": None,
                                "function_evaluations": None,
                                "constraint_violation": None,
                                "fixed_point_residual": None,
                                "notes": f"instance build failed: {build_error}",
                            }
                        else:
                            try:
                                if experiment == "prox":
                                    details = _solve_prox(instance, method, args)
                                elif experiment == "relaxation":
                                    details = _solve_relaxation_run(instance, method, args)
                                else:
                                    details = _solve_bnb_run(instance, method, args)
                            except Exception as error:
                                details = {
                                    "status": "error",
                                    "elapsed_seconds": math.nan,
                                    "solver_seconds": math.nan,
                                    "objective": None,
                                    "iterations": None,
                                    "function_evaluations": None,
                                    "constraint_violation": None,
                                    "fixed_point_residual": None,
                                    "notes": f"{type(error).__name__}: {error}",
                                }
                        row = {**base, **details}
                        existing[key] = row
                        fresh_rows.append(row)
                        _write_rows(args.output, list(existing.values()))
                        progress.update(1)
                        progress.set_postfix(
                            {
                                "scenario": scenario,
                                "exp": experiment,
                                "n": dimension,
                                "seed": seed_index + 1,
                                "method": method,
                                "status": row["status"],
                            }
                        )
                seed_bar.close()
            experiment_bar.close()
    progress.close()

    all_rows = list(existing.values())
    all_rows.sort(
        key=lambda row: (
            row["scenario"],
            row["experiment"],
            int(row["dimension"]),
            int(row.get("seed_index", 0)),
            METHODS.index(row["method"]),
        )
    )
    _write_rows(args.output, all_rows)
    summary_rows = _aggregate_rows(all_rows)
    summary_path = args.output.with_name(args.output.stem + "_summary.csv")
    _write_summary_rows(summary_path, summary_rows)
    _plot(args.plot, summary_rows)
    print(f"CSV: {args.output.resolve()}")
    print(f"Summary CSV: {summary_path.resolve()}")
    print(f"Plot: {args.plot.resolve()}")
    print(f"Rows written this run: {len(fresh_rows)}")
    if args.verify_accuracy:
        accuracy_rows, accuracy_summary = _run_accuracy_verification(args, dataset)
        _write_dict_rows(args.accuracy_output, accuracy_rows)
        _write_dict_rows(args.accuracy_summary_output, accuracy_summary)
        print(f"Accuracy CSV: {args.accuracy_output.resolve()}")
        print(f"Accuracy summary CSV: {args.accuracy_summary_output.resolve()}")
        _print_accuracy_summary(accuracy_summary)


if __name__ == "__main__":
    main()
