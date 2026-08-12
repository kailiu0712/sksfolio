"""Six-panel scaling benchmark for prox, relaxation, and exact sparse solves.

The script produces two rows of scaling plots:

1. a small-k setting;
2. a larger-k setting.

Each row contains three experiments:

- constrained proximal subproblem solves;
- continuous relaxation solves;
- exact sparse solves.

The four plotted method labels are:

- ``fista_dual_fista``;
- ``fista_dual_lbfgs``;
- ``gurobi``;
- ``mosek``.

For the exact sparse experiment, the first two labels denote the custom
certificate-driven BnB solver with the corresponding FISTA root-relaxation
oracle, while ``gurobi`` and ``mosek`` denote the native mixed-integer exact
references.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
from scipy import sparse
from tqdm.auto import tqdm

from ..bnb import solve_bnb
from ..incumbent import (
    solve_gurobi_incumbent,
    solve_incumbent,
    solve_mosek_incumbent,
)
from ..relaxation import solve_relaxation
from ..relaxation.fista import LinearConstraintProx
from .instance_generator import factor_operator_norm_squared, generate_instance


METHODS = (
    "fista_dual_fista",
    "fista_dual_lbfgs",
    "gurobi",
    "mosek",
)
EXPERIMENTS = ("prox", "relaxation", "bnb")
DEFAULT_DIMENSIONS = (
    10,
    20,
    30,
    50,
    80,
    100,
    150,
    200,
    300,
    500,
    800,
    1000,
    1500,
    2000,
    3000,
    4000,
    5000,
)
DEFAULT_PLOT = Path("benchmarks/four_scaling_six_panel.svg")
DEFAULT_RESULTS = Path("benchmarks/four_scaling_results.csv")
DEFAULT_SEED_COUNT = 10


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[7]


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _bootstrap_commercial_environment() -> dict[str, Any]:
    root = _workspace_root()
    info: dict[str, Any] = {}

    gurobi_license = root / "gurobi.lic"
    if gurobi_license.exists() and "GRB_LICENSE_FILE" not in os.environ:
        os.environ["GRB_LICENSE_FILE"] = str(gurobi_license)
        info["gurobi_license"] = str(gurobi_license)
    else:
        info["gurobi_license"] = os.environ.get("GRB_LICENSE_FILE")

    mosek_license = root / "mosek.lic"
    if mosek_license.exists() and "MOSEKLM_LICENSE_FILE" not in os.environ:
        os.environ["MOSEKLM_LICENSE_FILE"] = str(mosek_license)
        info["mosek_license"] = str(mosek_license)
    else:
        info["mosek_license"] = os.environ.get("MOSEKLM_LICENSE_FILE")

    if "mosek" not in sys.modules:
        for candidate in (
            root / "proximal" / "code0725" / ".venv" / "Lib" / "site-packages",
            root / "proximal" / ".venv" / "Lib" / "site-packages",
        ):
            if candidate.exists():
                candidate_text = str(candidate)
                if candidate_text not in sys.path:
                    sys.path.insert(0, candidate_text)
                try:
                    importlib.import_module("mosek")
                except Exception:
                    continue
                info["mosek_site_packages"] = candidate_text
                break
    return info


def _commercial_solver(backend: str) -> Callable[..., Mapping[str, Any]]:
    family, language = backend.split(".", 1)
    module = importlib.import_module(
        f"sksfolio.relaxation.{family}.{language}"
    )
    return module.solve


def _small_k(dimension: int) -> int:
    return min(dimension, max(2, min(10, int(round(0.02 * dimension)))))


def _large_k(dimension: int) -> int:
    small = _small_k(dimension)
    candidate = max(small + 1, int(round(0.05 * dimension)))
    return min(dimension, max(small + 1, min(50, candidate)))


SCENARIOS: tuple[tuple[str, Callable[[int], int]], ...] = (
    ("small_k", _small_k),
    ("large_k", _large_k),
)


def _rank_for_dimension(dimension: int, cap: int) -> int:
    return min(cap, max(2, dimension // 2))


def _constraint_counts(dimension: int) -> tuple[int, int, int]:
    sectors = min(10, max(2, dimension // 2))
    styles = min(4, max(1, dimension // 5))
    stresses = min(4, max(1, dimension // 5))
    return sectors, styles, stresses


def _build_instance(
    dimension: int,
    k: int,
    seed: int,
    rank_cap: int,
) -> Any:
    rank = _rank_for_dimension(dimension, rank_cap)
    sectors, styles, stresses = _constraint_counts(dimension)
    return generate_instance(
        dimension=dimension,
        rank=rank,
        k=k,
        gamma_scale=100.0,
        regime="hybrid",
        seed=seed,
        sectors=sectors,
        style_factors=styles,
        stress_constraints=stresses,
        target_fraction=0.0,
        target_iterations=200,
        sector_band=1.0,
        style_band=4.0,
        stress_band=4.0,
        annual_volatility=0.20,
        common_correlation=0.15,
    )


def _prox_problem_data(instance: Any) -> tuple[np.ndarray, float]:
    anchor = np.asarray(instance.anchor, dtype=float).reshape(-1)
    gradient = (
        np.asarray(instance.B @ (instance.B.T @ anchor), dtype=float).reshape(-1)
        - float(instance.return_reward) * np.asarray(instance.mu, dtype=float)
    )
    lipschitz = max(
        float(factor_operator_norm_squared(np.asarray(instance.B, dtype=float))),
        1e-12,
    )
    step = 1.0 / lipschitz
    argument = anchor - step * gradient
    gamma = step * float(instance.perspective_weight)
    return argument, gamma


def _prox_instance(instance: Any, argument: np.ndarray, gamma: float) -> dict[str, Any]:
    dimension = int(instance.dimension)
    return {
        "B": sparse.eye(dimension, format="csc", dtype=np.float64),
        "mu": np.asarray(argument, dtype=np.float64).reshape(-1),
        "C": instance.C,
        "lower": np.asarray(instance.lower, dtype=np.float64).reshape(-1),
        "upper": np.asarray(instance.upper, dtype=np.float64).reshape(-1),
        "anchor": np.asarray(instance.anchor, dtype=np.float64).reshape(-1),
        "constraint_names": tuple(instance.constraint_names),
        "k": int(instance.k),
        "perspective_weight": float(gamma),
        "return_reward": 1.0,
    }


def _row_base(
    scenario: str,
    experiment: str,
    method: str,
    dimension: int,
    k: int,
    rank: int,
    seed: int,
    seed_index: int,
) -> dict[str, Any]:
    return {
        "scenario": scenario,
        "experiment": experiment,
        "method": method,
        "dimension": int(dimension),
        "k": int(k),
        "rank": int(rank),
        "seed": int(seed),
        "seed_index": int(seed_index),
    }


def _solve_prox(
    instance: Any,
    method: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    argument, gamma = _prox_problem_data(instance)
    start = time.perf_counter()
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
        elapsed = time.perf_counter() - start
        return {
            "status": "converged" if result.converged else "iteration_limit",
            "elapsed_seconds": elapsed,
            "solver_seconds": elapsed,
            "objective": None,
            "iterations": result.iterations,
            "function_evaluations": result.function_evaluations,
            "constraint_violation": result.constraint_violation,
            "fixed_point_residual": result.fixed_point_residual,
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
    result = _commercial_solver(backend)(_prox_instance(instance, argument, gamma), options=options)
    elapsed = time.perf_counter() - start
    return {
        "status": result.get("status", "unknown"),
        "elapsed_seconds": elapsed,
        "solver_seconds": result.get("total_seconds", result.get("solve_seconds")),
        "objective": result.get("objective"),
        "iterations": result.get("iterations"),
        "function_evaluations": None,
        "constraint_violation": None,
        "fixed_point_residual": None,
        "notes": result.get("message", result.get("error")),
    }


def _solve_relaxation_run(
    instance: Any,
    method: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if method == "fista_dual_fista":
        backend = "fista"
        options = {
            "threads": args.threads,
            "max_iterations": args.relaxation_max_iterations,
            "time_limit": args.relaxation_time_limit,
            "tolerance": args.relaxation_tolerance,
            "feasibility_tolerance": args.relaxation_tolerance,
            "history_interval": args.relaxation_history_interval,
            "prox_tolerance": args.prox_tolerance,
            "prox_max_iterations": args.prox_max_iterations,
            "prox_oracle": "dual_fista",
            "restart_strategy": "gradient",
        }
    elif method == "fista_dual_lbfgs":
        backend = "fista"
        options = {
            "threads": args.threads,
            "max_iterations": args.relaxation_max_iterations,
            "time_limit": args.relaxation_time_limit,
            "tolerance": args.relaxation_tolerance,
            "feasibility_tolerance": args.relaxation_tolerance,
            "history_interval": args.relaxation_history_interval,
            "prox_tolerance": args.prox_tolerance,
            "prox_max_iterations": args.prox_max_iterations,
            "prox_oracle": "dual_lbfgs",
            "prox_lbfgs_memory": args.prox_lbfgs_memory,
            "prox_lbfgs_max_line_search": args.prox_lbfgs_max_line_search,
            "prox_lbfgs_fallback": args.prox_lbfgs_fallback,
            "restart_strategy": "gradient",
        }
    else:
        backend = "gurobi" if method == "gurobi" else "mosek"
        options = {
            "threads": args.threads,
            "time_limit": args.relaxation_time_limit,
            "tolerance": args.relaxation_tolerance,
            "log": False,
            "warm_start": False,
        }

    start = time.perf_counter()
    result = solve_relaxation(
        instance,
        backend=backend,
        warm_start=False,
        options=options,
    )
    elapsed = time.perf_counter() - start
    raw = result.raw
    return {
        "status": result.status,
        "elapsed_seconds": elapsed,
        "solver_seconds": raw.get(
            "total_seconds",
            raw.get("wrapper_seconds", raw.get("solve_seconds")),
        ),
        "objective": result.objective,
        "iterations": raw.get("iterations"),
        "function_evaluations": raw.get("prox_function_evaluations"),
        "constraint_violation": raw.get("violation"),
        "fixed_point_residual": raw.get("residual"),
        "notes": raw.get("prox_oracle_used", raw.get("error")),
    }


def _fista_relaxation_options(
    method: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    options = {
        "threads": args.threads,
        "max_iterations": args.relaxation_max_iterations,
        "time_limit": args.relaxation_time_limit,
        "tolerance": args.relaxation_tolerance,
        "feasibility_tolerance": args.relaxation_tolerance,
        "history_interval": args.relaxation_history_interval,
        "prox_tolerance": args.prox_tolerance,
        "prox_max_iterations": args.prox_max_iterations,
        "restart_strategy": "gradient",
    }
    if method == "fista_dual_fista":
        options["prox_oracle"] = "dual_fista"
        return options
    if method == "fista_dual_lbfgs":
        options.update(
            {
                "prox_oracle": "dual_lbfgs",
                "prox_lbfgs_memory": args.prox_lbfgs_memory,
                "prox_lbfgs_max_line_search": args.prox_lbfgs_max_line_search,
                "prox_lbfgs_fallback": args.prox_lbfgs_fallback,
            }
        )
        return options
    raise ValueError(f"unsupported FISTA relaxation method: {method}")


def _build_exact_warm_start(
    instance: Any,
    args: argparse.Namespace,
) -> tuple[Optional[Any], Optional[Any], Optional[str]]:
    root_method = str(args.commercial_exact_warm_start_root)
    fallback_method = (
        "fista_dual_fista"
        if root_method == "fista_dual_lbfgs"
        else "fista_dual_lbfgs"
    )
    for method in (root_method, fallback_method):
        try:
            relaxation = solve_relaxation(
                instance,
                backend="fista",
                warm_start=False,
                options=_fista_relaxation_options(method, args),
            )
            incumbent = solve_incumbent(
                instance,
                relaxation,
                method="auto",
                restricted_solver="osqp",
                time_limit=args.incumbent_time_limit,
                random_state=args.seed,
            )
        except Exception:
            continue
        if incumbent.feasible:
            return relaxation, incumbent, method
    return None, None, None


def _solve_bnb_run(
    instance: Any,
    method: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    start = time.perf_counter()
    if method in {"fista_dual_fista", "fista_dual_lbfgs"}:
        relaxation = solve_relaxation(
            instance,
            backend="fista",
            warm_start=False,
            options=_fista_relaxation_options(method, args),
        )
        incumbent = solve_incumbent(
            instance,
            relaxation,
            method="auto",
            restricted_solver="osqp",
            time_limit=args.incumbent_time_limit,
            random_state=args.seed,
        )
        result = solve_bnb(
            instance,
            relaxation=relaxation,
            incumbent=(incumbent if incumbent.feasible else None),
            restricted_solver="osqp",
            time_limit=args.bnb_time_limit,
            node_limit=args.bnb_node_limit,
            relative_gap=args.bnb_relative_gap,
            absolute_gap=args.bnb_absolute_gap,
            options={
                "safe_screening": True,
                "multi_selector_cuts": False,
                "root_pair_cuts": False,
                "branching_rule": "max_min",
                "node_dual_iterations": args.node_dual_iterations,
                "node_heuristic_frequency": 1,
                "restricted_qp_options": {
                    "time_limit": args.bnb_time_limit,
                },
            },
        )
        elapsed = time.perf_counter() - start
        return {
            "status": result.status,
            "elapsed_seconds": elapsed,
            "solver_seconds": result.raw.get("solve_seconds"),
            "objective": result.upper_bound,
            "iterations": result.raw.get("nodes_processed"),
            "function_evaluations": result.raw.get("node_dual_function_evaluations"),
            "constraint_violation": None,
            "fixed_point_residual": None,
            "notes": result.raw.get("best_incumbent_source"),
        }

    warm_start = None
    warm_start_method = None
    if args.commercial_exact_warm_start:
        _, warm_start, warm_start_method = _build_exact_warm_start(
            instance,
            args,
        )

    if method == "gurobi":
        result = solve_gurobi_incumbent(
            instance,
            warm_start=warm_start,
            time_limit=args.exact_time_limit,
            options={
                "verbose": False,
                "Threads": args.threads,
                "MIPFocus": 0,
                "MIPGap": args.bnb_relative_gap,
                "MIPGapAbs": args.bnb_absolute_gap,
                "FeasibilityTol": 1e-8,
                "OptimalityTol": 1e-8,
                "BarQCPConvTol": 1e-8,
                "Seed": args.seed,
            },
        )
    else:
        result = solve_mosek_incumbent(
            instance,
            warm_start=warm_start,
            time_limit=args.exact_time_limit,
            options={
                "verbose": False,
                "threads": args.threads,
                "relative_gap": args.bnb_relative_gap,
                "absolute_gap": args.bnb_absolute_gap,
            },
        )
    elapsed = time.perf_counter() - start
    raw = result.raw
    return {
        "status": result.status,
        "elapsed_seconds": elapsed,
        "solver_seconds": raw.get("total_seconds", raw.get("solve_seconds")),
        "objective": result.upper_bound,
        "iterations": raw.get("node_count"),
        "function_evaluations": raw.get("solution_count"),
        "constraint_violation": None,
        "fixed_point_residual": None,
        "notes": (
            raw.get("formulation", raw.get("error"))
            if warm_start_method is None
            else f"{raw.get('formulation', raw.get('error'))}; warm_start={warm_start_method}"
        ),
    }


def _seed_values(
    base_seed: int,
    scenario: str,
    dimension: int,
    count: int,
) -> tuple[int, ...]:
    scenario_offset = 0 if scenario == "small_k" else 1_000_000
    start = int(base_seed + scenario_offset + 10_000 * int(dimension))
    return tuple(start + offset for offset in range(int(count)))


def _existing_rows(
    path: Path,
) -> dict[tuple[str, str, int, str, int], dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = {}
        for row in reader:
            key = (
                str(row["scenario"]),
                str(row["experiment"]),
                int(row["dimension"]),
                str(row["method"]),
                int(row.get("seed_index", 0)),
            )
            rows[key] = row
        return rows


def _write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario",
        "experiment",
        "method",
        "dimension",
        "k",
        "rank",
        "seed",
        "seed_index",
        "status",
        "elapsed_seconds",
        "solver_seconds",
        "objective",
        "iterations",
        "function_evaluations",
        "constraint_violation",
        "fixed_point_residual",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["scenario"]),
            str(row["experiment"]),
            str(row["method"]),
            int(row["dimension"]),
        )
        grouped.setdefault(key, []).append(row)

    summaries: list[dict[str, Any]] = []
    for key, members in grouped.items():
        scenario, experiment, method, dimension = key
        elapsed = [
            float(row["elapsed_seconds"])
            for row in members
            if row.get("elapsed_seconds") not in {None, ""}
            and math.isfinite(float(row["elapsed_seconds"]))
        ]
        solver = [
            float(row["solver_seconds"])
            for row in members
            if row.get("solver_seconds") not in {None, ""}
            and math.isfinite(float(row["solver_seconds"]))
        ]
        summaries.append(
            {
                "scenario": scenario,
                "experiment": experiment,
                "method": method,
                "dimension": dimension,
                "k": int(members[0]["k"]),
                "rank": int(members[0]["rank"]),
                "elapsed_seconds": (
                    float(np.mean(elapsed)) if elapsed else math.nan
                ),
                "elapsed_seconds_std": (
                    float(np.std(elapsed)) if elapsed else math.nan
                ),
                "solver_seconds": (
                    float(np.mean(solver)) if solver else math.nan
                ),
                "solver_seconds_std": (
                    float(np.std(solver)) if solver else math.nan
                ),
                "valid_elapsed_runs": len(elapsed),
                "seed_runs": len(members),
                "status_counts": "|".join(
                    f"{status}:{count}"
                    for status, count in sorted(
                        {
                            status: sum(
                                1
                                for row in members
                                if str(row["status"]) == status
                            )
                            for status in {
                                str(row["status"]) for row in members
                            }
                        }.items()
                    )
                ),
            }
        )
    summaries.sort(
        key=lambda row: (
            row["scenario"],
            row["experiment"],
            int(row["dimension"]),
            METHODS.index(str(row["method"])),
        )
    )
    return summaries


def _write_summary_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario",
        "experiment",
        "method",
        "dimension",
        "k",
        "rank",
        "elapsed_seconds",
        "elapsed_seconds_std",
        "solver_seconds",
        "solver_seconds_std",
        "valid_elapsed_runs",
        "seed_runs",
        "status_counts",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {
        "fista_dual_fista": {"color": "#1f77b4", "marker": "o"},
        "fista_dual_lbfgs": {"color": "#d62728", "marker": "s"},
        "gurobi": {"color": "#2ca02c", "marker": "^"},
        "mosek": {"color": "#ff7f0e", "marker": "D"},
    }
    titles = {
        ("small_k", "prox"): "Small k: Proximal Solve",
        ("small_k", "relaxation"): "Small k: Relaxation Solve",
        ("small_k", "bnb"): "Small k: Exact Sparse Solve",
        ("large_k", "prox"): "Large k: Proximal Solve",
        ("large_k", "relaxation"): "Large k: Relaxation Solve",
        ("large_k", "bnb"): "Large k: Exact Sparse Solve",
    }
    figure, axes = plt.subplots(2, 3, figsize=(20, 10), constrained_layout=False)
    figure.subplots_adjust(
        left=0.055,
        right=0.99,
        bottom=0.07,
        top=0.88,
        wspace=0.18,
        hspace=0.28,
    )
    axes_map = {
        ("small_k", "prox"): axes[0, 0],
        ("small_k", "relaxation"): axes[0, 1],
        ("small_k", "bnb"): axes[0, 2],
        ("large_k", "prox"): axes[1, 0],
        ("large_k", "relaxation"): axes[1, 1],
        ("large_k", "bnb"): axes[1, 2],
    }
    handles = []
    labels = []

    for key, axis in axes_map.items():
        scenario, experiment = key
        axis.set_title(titles[key])
        axis.set_xlabel("Dimension n")
        axis.set_ylabel("Mean Elapsed Seconds")
        axis.set_xlim(10, 5000)
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.25)

        subset = [
            row for row in rows
            if row["scenario"] == scenario and row["experiment"] == experiment
        ]
        for method in METHODS:
            method_rows = [
                row for row in subset
                if row["method"] == method
                and row.get("elapsed_seconds") is not None
                and math.isfinite(float(row["elapsed_seconds"]))
            ]
            method_rows.sort(key=lambda row: int(row["dimension"]))
            if not method_rows:
                continue
            x = [int(row["dimension"]) for row in method_rows]
            y = [float(row["elapsed_seconds"]) for row in method_rows]
            line, = axis.plot(
                x,
                y,
                linewidth=2.0,
                markersize=5.0,
                label=method,
                **styles[method],
            )
            if method not in labels:
                handles.append(line)
                labels.append(method)

    if handles:
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.965),
            ncol=4,
            frameon=False,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    plt.close(figure)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Run six scaling plots for prox, relaxation, and exact sparse "
            "solves under small-k and larger-k settings"
        )
    )
    result.add_argument(
        "--dimensions",
        type=int,
        nargs="+",
        default=list(DEFAULT_DIMENSIONS),
        help="sampled dimensions spanning n=10 to n=5000",
    )
    result.add_argument("--seed", type=int, default=17)
    result.add_argument("--seed-count", type=int, default=DEFAULT_SEED_COUNT)
    result.add_argument("--rank-cap", type=int, default=50)
    result.add_argument("--threads", type=int, default=1)
    result.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS,
    )
    result.add_argument(
        "--plot",
        type=Path,
        default=DEFAULT_PLOT,
    )
    result.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    result.add_argument("--prox-tolerance", type=float, default=1e-8)
    result.add_argument("--prox-max-iterations", type=int, default=2_000)
    result.add_argument("--prox-time-limit", type=float, default=15.0)
    result.add_argument("--prox-lbfgs-memory", type=int, default=10)
    result.add_argument("--prox-lbfgs-max-line-search", type=int, default=40)
    result.add_argument(
        "--prox-lbfgs-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    result.add_argument("--relaxation-tolerance", type=float, default=1e-7)
    result.add_argument(
        "--relaxation-max-iterations",
        type=int,
        default=10_000,
    )
    result.add_argument("--relaxation-time-limit", type=float, default=20.0)
    result.add_argument("--relaxation-history-interval", type=int, default=25)
    result.add_argument("--incumbent-time-limit", type=float, default=5.0)
    result.add_argument("--bnb-time-limit", type=float, default=10.0)
    result.add_argument("--exact-time-limit", type=float, default=10.0)
    result.add_argument("--bnb-node-limit", type=int, default=200_000)
    result.add_argument("--bnb-relative-gap", type=float, default=1e-4)
    result.add_argument("--bnb-absolute-gap", type=float, default=1e-8)
    result.add_argument("--node-dual-iterations", type=int, default=10)
    result.add_argument(
        "--commercial-exact-warm-start",
        action="store_true",
        help="warm-start Gurobi and MOSEK exact solves with a shared OSQP incumbent",
    )
    result.add_argument(
        "--commercial-exact-warm-start-root",
        choices=("fista_dual_lbfgs", "fista_dual_fista"),
        default="fista_dual_lbfgs",
        help="FISTA root relaxation used to build the commercial exact warm start",
    )
    return result


def main(arguments: Optional[Sequence[str]] = None) -> None:
    args = parser().parse_args(arguments)
    env_info = _bootstrap_commercial_environment()

    print("Four-scaling benchmark")
    print(f"Workspace root: {_workspace_root()}")
    print(f"Repository root: {_repository_root()}")
    print(f"Gurobi license: {env_info.get('gurobi_license')}")
    print(f"MOSEK license: {env_info.get('mosek_license')}")
    print(f"MOSEK site-packages: {env_info.get('mosek_site_packages')}")
    print(f"Dimensions: {tuple(args.dimensions)}")
    print(f"Seeds per dimension: {args.seed_count}")

    existing = _existing_rows(args.output) if args.resume else {}
    keys_to_run: list[tuple[str, str, int, str, int]] = []
    for scenario, _ in SCENARIOS:
        for experiment in EXPERIMENTS:
            for dimension in args.dimensions:
                seeds = _seed_values(
                    args.seed,
                    scenario,
                    int(dimension),
                    args.seed_count,
                )
                for seed_index, _ in enumerate(seeds):
                    for method in METHODS:
                        key = (
                            scenario,
                            experiment,
                            int(dimension),
                            method,
                            int(seed_index),
                        )
                        if key not in existing:
                            keys_to_run.append(key)

    print(f"Cached rows reused: {len(existing)}")
    print(f"Rows to run now: {len(keys_to_run)}")
    progress = tqdm(total=len(keys_to_run), desc="four-scaling", unit="run")
    fresh_rows: list[dict[str, Any]] = []

    for scenario, k_rule in SCENARIOS:
        print(f"[scenario] {scenario}")
        for experiment in EXPERIMENTS:
            print(f"  [experiment] {experiment}")
            experiment_bar = tqdm(
                [dimension for dimension in args.dimensions],
                desc=f"{scenario}:{experiment}",
                leave=False,
                unit="n",
            )
            for dimension in experiment_bar:
                k = k_rule(int(dimension))
                seeds = _seed_values(
                    args.seed,
                    scenario,
                    int(dimension),
                    args.seed_count,
                )
                experiment_bar.set_postfix({"n": dimension, "k": k})
                seed_bar = tqdm(
                    list(enumerate(seeds)),
                    desc=f"{scenario}:{experiment}:n{dimension}",
                    leave=False,
                    unit="seed",
                )
                for seed_index, seed in seed_bar:
                    instance = _build_instance(
                        int(dimension),
                        k,
                        int(seed),
                        int(args.rank_cap),
                    )
                    seed_bar.set_postfix(
                        {"seed": seed_index + 1, "of": len(seeds)}
                    )
                    for method in METHODS:
                        key = (
                            scenario,
                            experiment,
                            int(dimension),
                            method,
                            int(seed_index),
                        )
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
                            int(instance.rank),
                            int(seed),
                            int(seed_index),
                        )
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


if __name__ == "__main__":
    main()
