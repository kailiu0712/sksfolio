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

if __package__ in {None, ""}:
    _PACKAGE_ROOT = Path(__file__).resolve().parents[2]
    if str(_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_ROOT))
    from sksfolio.bnb import solve_bnb
    from sksfolio.incumbent import (
        solve_gurobi_incumbent,
        solve_incumbent,
        solve_mosek_incumbent,
    )
    from sksfolio.relaxation import solve_relaxation
    from sksfolio.relaxation.fista import LinearConstraintProx
    from sksfolio.benchmarks.bertsimas_cory_wright import (
        CONSTRAINT_PROFILES,
    )
    from sksfolio.benchmarks.instance_generator import (
        factor_operator_norm_squared,
        generate_instance,
    )
else:
    from ..bnb import solve_bnb
    from ..incumbent import (
        solve_gurobi_incumbent,
        solve_incumbent,
        solve_mosek_incumbent,
    )
    from ..relaxation import solve_relaxation
    from ..relaxation.fista import LinearConstraintProx
    from .bertsimas_cory_wright import CONSTRAINT_PROFILES
    from .instance_generator import (
        factor_operator_norm_squared,
        generate_instance,
    )



RESUME_FROM_OUTPUT = "@output"
METHODS = (
    "fista_dual_fista",
    "fista_dual_lbfgs",
    "hybrid_newton",
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
DEFAULT_SEED_COUNT = 10
# Exact-solve ladder.  Every method exhausts its time limit from about
# n=1500 upward, so those rungs all report the same saturation at roughly
# 143 seconds per dimension and seed.  Keeping full resolution up to 1000
# and two anchors above it costs about an hour instead of three and a half
# while showing the same thing.
DEFAULT_EXACT_DIMENSIONS = (
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
    2000,
    5000,
)
DEFAULT_EXACT_SEED_COUNT = 5
BCW_STANDARD_PROFILE = dict(CONSTRAINT_PROFILES["standard"])


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[6]


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _results_root() -> Path:
    return _workspace_root() / "proximal" / "code0821brian" / "results"


def _default_result_path(filename: str) -> Path:
    return _results_root() / filename


DEFAULT_PLOT = _default_result_path("four_scaling_six_panel.svg")
DEFAULT_RESULTS = _default_result_path("four_scaling_results.csv")
DEFAULT_PRECISION_PLOT = _default_result_path("four_scaling_precision.svg")


def _bootstrap_commercial_environment(
    gurobi_license_mode: str = "file",
) -> dict[str, Any]:
    root = _workspace_root()
    info: dict[str, Any] = {}

    gurobi_license = root / "gurobi.lic"
    mode = str(gurobi_license_mode).strip().lower()
    if mode not in {"auto", "default", "file"}:
        raise ValueError("gurobi_license_mode must be auto, default, or file")
    if mode == "file":
        if not gurobi_license.exists():
            raise FileNotFoundError(f"Gurobi license file not found: {gurobi_license}")
        os.environ["GRB_LICENSE_FILE"] = str(gurobi_license)
        info["gurobi_license"] = str(gurobi_license)
    elif mode == "default":
        os.environ.pop("GRB_LICENSE_FILE", None)
        info["gurobi_license"] = "default"
    elif "GRB_LICENSE_FILE" in os.environ:
        info["gurobi_license"] = os.environ.get("GRB_LICENSE_FILE")
    else:
        default_gurobi_ok = False
        try:
            import gurobipy as gp

            model = gp.Model()
            model.Params.OutputFlag = 0
            model.dispose()
            default_gurobi_ok = True
        except Exception:
            default_gurobi_ok = False
        if default_gurobi_ok:
            info["gurobi_license"] = "default"
        elif gurobi_license.exists():
            os.environ["GRB_LICENSE_FILE"] = str(gurobi_license)
            info["gurobi_license"] = str(gurobi_license)
        else:
            info["gurobi_license"] = None

    mosek_license = root / "mosek.lic"
    if mosek_license.exists() and "MOSEKLM_LICENSE_FILE" not in os.environ:
        os.environ["MOSEKLM_LICENSE_FILE"] = str(mosek_license)
        info["mosek_license"] = str(mosek_license)
    else:
        info["mosek_license"] = os.environ.get("MOSEKLM_LICENSE_FILE")

    if "mosek" not in sys.modules:
        # Prefer a MOSEK that is already importable.  The fallback below
        # prepends an unrelated virtual environment to sys.path, which then
        # wins for *every* later import, not just mosek: that environment
        # ships numpy 2.5 / scipy 1.18 / osqp 1.1.3 binaries, and loading its
        # osqp into an interpreter running numpy 2.0 / scipy 1.14 aborts the
        # process with an access violation once pyarrow is also loaded.  That
        # is what killed the real-data run at its first branch-and-bound
        # solve.  Only reach for the fallback when there is no other MOSEK.
        try:
            importlib.import_module("mosek")
            info["mosek_site_packages"] = "interpreter"
        except Exception:
            for candidate in (
                root / "proximal" / "code0725" / ".venv" / "Lib" / "site-packages",
                root / "proximal" / ".venv" / "Lib" / "site-packages",
            ):
                if candidate.exists():
                    candidate_text = str(candidate)
                    if candidate_text not in sys.path:
                        sys.path.append(candidate_text)
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


def _corrected_relaxation_backend(method: str) -> str:
    if method == "hybrid_newton":
        return "hybrid_newton"
    if method == "fista_dual_fista":
        return "corrected_fista"
    if method == "fista_dual_lbfgs":
        return "corrected_lbfgs"
    raise ValueError(f"unsupported corrected method: {method}")


def _sector_count(dimension: int) -> int:
    return min(20, max(2, int(round(dimension / 50.0))))


def _small_k(dimension: int) -> int:
    """Smallest cardinality budget that leaves the instance feasible.

    Every sector row carries a strictly positive lower bound on a support
    disjoint from the other sectors, so a portfolio holding fewer assets
    than there are sectors cannot fund them all and the exact problem is
    empty regardless of the data.  A budget below the sector count therefore
    measures infeasibility detection rather than optimization, which is what
    the earlier fixed cap of 10 did for every dimension above 500.  Two
    assets of slack keep the search non-trivial without making it vacuous.
    """
    return min(dimension, _sector_count(dimension) + 2)


def _large_k(dimension: int) -> int:
    """Mirror of :func:`_small_k` with a threefold sector allowance."""
    small = _small_k(dimension)
    return min(dimension, max(small + 1, 3 * _sector_count(dimension) + 2))


SCENARIOS: tuple[tuple[str, Callable[[int], int]], ...] = (
    ("small_k", _small_k),
    ("large_k", _large_k),
)


def _rank_for_dimension(dimension: int, cap: int) -> int:
    paper_rank = 50 if dimension <= 1_000 else 100
    return min(dimension - 1, max(2, min(cap, paper_rank)))


def _constraint_counts(dimension: int) -> tuple[int, int, int]:
    sectors = _sector_count(dimension)
    styles = min(8, max(1, int(round(dimension / 125.0))))
    stresses = min(15, max(1, int(round(dimension / 67.0))))
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


def _prox_objective(
    instance: Any,
    argument: np.ndarray,
    gamma: float,
    x: Any,
) -> Optional[float]:
    """Score a proximal solution on the objective the backends all share."""
    if x is None:
        return None
    from sksfolio.relaxation.problem import (
        MarkowitzInstance,
        evaluate_solution,
    )

    data = _prox_instance(instance, argument, gamma)
    problem = MarkowitzInstance(
        factor_loadings=data["B"],
        expected_returns=data["mu"],
        constraint_matrix=data["C"],
        lower_bounds=data["lower"],
        upper_bounds=data["upper"],
        feasible_anchor=data["anchor"],
        constraint_names=list(data["constraint_names"]),
        k=data["k"],
        perspective_weight=data["perspective_weight"],
        return_reward=data["return_reward"],
        anchor_must_be_feasible=False,
    )
    try:
        return float(evaluate_solution(problem, x)["objective"])
    except Exception:
        return None


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
    if method in {"fista_dual_fista", "fista_dual_lbfgs", "hybrid_newton"}:
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
            semismooth_newton=method == "hybrid_newton",
        )
        result = oracle.solve(argument, gamma)
        elapsed = time.perf_counter() - start
        # Scored after the timer: the commercial backends report the same
        # proximal objective, so recording it here is what lets the accuracy
        # panel compare every method's prox solution against Gurobi's.
        return {
            "status": "converged" if result.converged else "iteration_limit",
            "elapsed_seconds": elapsed,
            "solver_seconds": elapsed,
            "objective": _prox_objective(instance, argument, gamma, result.x),
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
    # Score the returned point with the same function used for the custom
    # oracles rather than each backend's own objective report, so the
    # accuracy panel compares solutions and not bookkeeping conventions.
    return {
        "status": result.get("status", "unknown"),
        "elapsed_seconds": elapsed,
        "solver_seconds": result.get("total_seconds", result.get("solve_seconds")),
        "objective": _prox_objective(instance, argument, gamma, result.get("x")),
        "iterations": result.get("iterations"),
        "function_evaluations": None,
        "constraint_violation": result.get("violation"),
        "fixed_point_residual": None,
        "notes": result.get("message", result.get("error")),
    }


def _solve_relaxation_run(
    instance: Any,
    method: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if method in {"fista_dual_fista", "fista_dual_lbfgs", "hybrid_newton"}:
        backend = _corrected_relaxation_backend(method)
        options = _fista_relaxation_options(method, args)
    else:
        backend = "gurobi" if method == "gurobi" else "mosek"
        options = {
            "threads": args.threads,
            "time_limit": args.relaxation_time_limit,
            "tolerance": args.relaxation_tolerance,
            "log": False,
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
        return options
    if method in {"fista_dual_lbfgs", "hybrid_newton"}:
        options.update(
            {
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
                backend=_corrected_relaxation_backend(method),
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
    if method in {"fista_dual_fista", "fista_dual_lbfgs", "hybrid_newton"}:
        from sksfolio.bnb.cardinality_presolve import cardinality_certificate
        if cardinality_certificate(instance) is not None:
            result = solve_bnb(instance, time_limit=args.bnb_time_limit)
            return {
                "status": result.status,
                "elapsed_seconds": time.perf_counter()-start,
                "solver_seconds": result.raw.get("solve_seconds"),
                "objective": None, "iterations": 0,
                "function_evaluations": 0, "constraint_violation": None,
                "fixed_point_residual": None,
                "notes": result.raw.get("infeasibility_source"),
            }
        relaxation = solve_relaxation(
            instance,
            backend=_corrected_relaxation_backend(method),
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
                "polish_incumbent": False,
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


def _experiment_dimensions(
    experiment: str,
    dimensions: Sequence[int],
    exact_max_dimension: Optional[int],
    exact_dimensions: Optional[Sequence[int]] = None,
) -> list[int]:
    """Dimensions to run for one experiment.

    The exact sparse solve is the only experiment with its own ladder: past
    a few hundred assets every method exhausts its time limit, so those rows
    cost the full budget and report the cap rather than a solve time.  An
    explicit ``exact_dimensions`` list wins over the ceiling, which is how a
    run keeps a few large dimensions for coverage without paying for every
    rung between them.
    """
    kept = [int(value) for value in dimensions]
    if experiment != "bnb":
        return kept
    if exact_dimensions:
        # Intersected, not substituted: the exact ladder is a default, so a
        # run that narrows --dimensions for a quick check must not silently
        # pull the full ladder -- including its expensive top rungs -- back
        # into the exact experiment.
        wanted = {int(value) for value in exact_dimensions}
        return [value for value in kept if value in wanted]
    if exact_max_dimension is None:
        return kept
    return [value for value in kept if value <= int(exact_max_dimension)]


def _experiment_seeds(
    experiment: str,
    seeds: Sequence[int],
    exact_seed_count: Optional[int],
) -> list[int]:
    """Seeds to run for one experiment.

    The exact solve costs its full time limit on most instances, so it uses
    a prefix of the same seed list rather than a separate draw: its rows
    stay a subset of the prox and relaxation instances and remain directly
    comparable to them.
    """
    values = [int(seed) for seed in seeds]
    if experiment != "bnb" or exact_seed_count is None:
        return values
    return values[: max(1, int(exact_seed_count))]


def _existing_rows(
    path: Path,
) -> dict[tuple[str, str, int, str, int], dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[tuple[str, str, int, str, int], dict[str, Any]] = {}
    skipped = 0
    try:
        with path.open("r", newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            for row in reader:
                status = str(row.get("status", "")).strip().lower()
                if status in {"error", "unavailable"}:
                    continue
                try:
                    key = (
                        str(row["scenario"]),
                        str(row["experiment"]),
                        int(row["dimension"]),
                        str(row["method"]),
                        int(row.get("seed_index", 0)),
                    )
                except (KeyError, TypeError, ValueError):
                    # A row damaged by an interrupted write is dropped and
                    # recomputed rather than aborting the whole resume.
                    skipped += 1
                    continue
                rows[key] = row
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        print(f"checkpoint {path} is unreadable ({error}); starting fresh")
        return {}
    if skipped:
        print(f"checkpoint {path}: skipped {skipped} unreadable rows")
    return rows


def _resume_rows(
    resume: Optional[str],
    output: Path,
) -> tuple[
    Optional[Path],
    dict[tuple[str, str, int, str, int], dict[str, Any]],
]:
    """Resolve ``--resume`` into the checkpoint to reuse and its rows.

    ``None`` means a fresh run and is the default: every row is measured
    again, so a result file never silently mixes rows from different
    sessions.  Cached rows are only comparable with new ones when the same
    process produced both -- real-data instances in particular are not
    reproducible across processes, because ``svds`` is started from an
    unseeded vector.  Resuming therefore has to be asked for, and it names
    its checkpoint instead of inferring one from ``--output``.
    """
    if resume is None:
        return None, {}
    checkpoint = output if resume == RESUME_FROM_OUTPUT else Path(resume)
    return checkpoint, _existing_rows(checkpoint)


def _archive_previous_results(output: Path) -> None:
    """Move an earlier result file aside before a fresh run replaces it.

    A fresh run rewrites ``output`` as soon as its first row is solved, so
    rerunning with default paths would otherwise discard a previous
    multi-hour result before anyone could notice.
    """
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for path in (output, output.with_name(output.stem + "_summary.csv")):
        if not path.exists():
            continue
        archived = path.with_name(
            f"{path.stem}.superseded-{stamp}{path.suffix}"
        )
        os.replace(path, archived)
        print(f"Archived previous result: {path.name} -> {archived.name}")


def _atomic_write(path: Path, render: Callable[[Any], None]) -> None:
    """Write a checkpoint so a reader never observes a torn file.

    The content is rendered into a sibling ``.partial``, flushed, and fsynced
    before replacing the real file, so an interruption leaves either the
    previous complete checkpoint or the new one.  On Windows the replacement
    intermittently fails while a virus scanner or the search indexer still
    holds a handle on one of the two files, so it is retried before falling
    back to writing in place: losing a multi-hour run to a transient lock is
    worse than the short window the replacement protects.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", newline="", encoding="utf-8") as stream:
        render(stream)
        stream.flush()
        os.fsync(stream.fileno())
    for attempt in range(8):
        try:
            os.replace(partial, path)
            return
        except PermissionError:
            time.sleep(0.25 * (attempt + 1))
    with path.open("w", newline="", encoding="utf-8") as stream:
        render(stream)
        stream.flush()
        os.fsync(stream.fileno())
    partial.unlink(missing_ok=True)


def _write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
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

    def render(stream: Any) -> None:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    _atomic_write(path, render)


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

    def render(stream: Any) -> None:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    _atomic_write(path, render)


def _plot(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {
        "fista_dual_fista": {"color": "#1f77b4", "marker": "o"},
        "fista_dual_lbfgs": {"color": "#d62728", "marker": "s"},
        "hybrid_newton": {"color": "#9467bd", "marker": "P"},
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


SOLVED_STATUSES = {"converged", "optimal", "gap_limit"}
TIME_LIMIT_STATUSES = {"time_limit", "iteration_limit", "node_limit"}
INFEASIBLE_STATUSES = {
    "infeasible",
    "infeasible_or_unbounded",
    "no_solution",
}
REFERENCE_METHOD = "gurobi"


def _row_objective(row: Mapping[str, Any]) -> Optional[float]:
    value = row.get("objective")
    if value in (None, "", "None"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _accuracy_series(
    rows: Sequence[Mapping[str, Any]],
    scenario: str,
    experiment: str,
    method: str,
) -> tuple[list[int], list[float], list[float]]:
    """Relative objective disagreement with Gurobi on shared instances.

    Only instances where both the method and the reference reached a
    solution are scored: a capped or infeasible run has no objective worth
    comparing, and silently folding those in would report agreement that
    was never measured.
    """
    reference = {
        (int(row["dimension"]), int(row.get("seed_index", 0))): row
        for row in rows
        if row["scenario"] == scenario
        and row["experiment"] == experiment
        and row["method"] == REFERENCE_METHOD
    }
    grouped: dict[int, list[float]] = {}
    for row in rows:
        if (
            row["scenario"] != scenario
            or row["experiment"] != experiment
            or row["method"] != method
        ):
            continue
        key = (int(row["dimension"]), int(row.get("seed_index", 0)))
        peer = reference.get(key)
        if peer is None:
            continue
        if str(row["status"]) not in SOLVED_STATUSES:
            continue
        if str(peer["status"]) not in SOLVED_STATUSES:
            continue
        value = _row_objective(row)
        benchmark = _row_objective(peer)
        if value is None or benchmark is None:
            continue
        scale = max(abs(benchmark), 1e-12)
        grouped.setdefault(int(row["dimension"]), []).append(
            abs(value - benchmark) / scale
        )
    dimensions = sorted(grouped)
    median = [float(np.median(grouped[n])) for n in dimensions]
    worst = [float(np.max(grouped[n])) for n in dimensions]
    return dimensions, median, worst


def _outcome_fractions(
    rows: Sequence[Mapping[str, Any]],
    scenario: str,
    experiment: str,
    method: str,
) -> tuple[list[int], list[float], list[float], list[float]]:
    """Per-dimension share of runs that solved, hit a limit, or were empty."""
    grouped: dict[int, list[str]] = {}
    for row in rows:
        if (
            row["scenario"] == scenario
            and row["experiment"] == experiment
            and row["method"] == method
        ):
            grouped.setdefault(int(row["dimension"]), []).append(
                str(row["status"])
            )
    dimensions = sorted(grouped)
    solved: list[float] = []
    capped: list[float] = []
    empty: list[float] = []
    for n in dimensions:
        statuses = grouped[n]
        total = float(len(statuses))
        solved.append(
            sum(s in SOLVED_STATUSES for s in statuses) / total
        )
        capped.append(
            sum(s in TIME_LIMIT_STATUSES for s in statuses) / total
        )
        empty.append(
            sum(s in INFEASIBLE_STATUSES for s in statuses) / total
        )
    return dimensions, solved, capped, empty


def _precision_plot(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Plot solution agreement with Gurobi and per-dimension outcome shares.

    The timing figure answers "how fast" only for runs that actually solved
    the problem to the requested tolerance.  This companion answers the two
    questions that have to be settled before a timing curve means anything:
    is the answer right, and what share of runs produced an answer at all.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {
        "fista_dual_fista": {"color": "#1f77b4", "marker": "o"},
        "fista_dual_lbfgs": {"color": "#d62728", "marker": "s"},
        "hybrid_newton": {"color": "#9467bd", "marker": "P"},
        "gurobi": {"color": "#2ca02c", "marker": "^"},
        "mosek": {"color": "#ff7f0e", "marker": "D"},
    }
    experiments = ("prox", "relaxation", "bnb")
    experiment_titles = {
        "prox": "Proximal Solve",
        "relaxation": "Relaxation Solve",
        "bnb": "Exact Sparse Solve",
    }
    figure, axes = plt.subplots(4, 3, figsize=(20, 19), constrained_layout=False)
    figure.subplots_adjust(
        left=0.055,
        right=0.99,
        bottom=0.05,
        top=0.90,
        wspace=0.2,
        hspace=0.36,
    )
    handles: list[Any] = []
    labels: list[str] = []
    # Outcome shares saturate at 0 and 1, where every method draws the same
    # line and only the last one painted stays visible.  A small constant
    # offset per method keeps all five readable; it is cosmetic, and the
    # axis is labelled as a fraction of runs.
    offsets = {
        method: (index - 0.5 * (len(METHODS) - 1)) * 0.013
        for index, method in enumerate(METHODS)
    }
    all_dimensions = sorted(
        {int(row["dimension"]) for row in rows}
    )
    span = (
        (all_dimensions[0] * 0.85, all_dimensions[-1] * 1.18)
        if all_dimensions
        else (10, 5000)
    )

    for scenario_index, scenario in enumerate(("small_k", "large_k")):
        for column, experiment in enumerate(experiments):
            accuracy_axis = axes[2 * scenario_index, column]
            outcome_axis = axes[2 * scenario_index + 1, column]

            accuracy_axis.set_title(
                f"{scenario}: {experiment_titles[experiment]} "
                "-- objective vs Gurobi"
            )
            accuracy_axis.set_xlabel("Dimension n")
            accuracy_axis.set_ylabel("Relative objective difference")
            accuracy_axis.set_xscale("log")
            accuracy_axis.set_yscale("log")
            accuracy_axis.set_xlim(*span)
            accuracy_axis.grid(True, which="both", alpha=0.25)
            accuracy_axis.axhline(
                1e-6,
                color="#444444",
                linestyle=":",
                linewidth=1.2,
            )

            outcome_axis.set_title(
                f"{scenario}: {experiment_titles[experiment]} "
                "-- outcome shares"
            )
            outcome_axis.set_xlabel("Dimension n")
            outcome_axis.set_ylabel("Fraction of runs")
            outcome_axis.set_xscale("log")
            outcome_axis.set_xlim(*span)
            outcome_axis.set_ylim(-0.09, 1.09)
            outcome_axis.grid(True, which="both", alpha=0.25)

            empty_drawn = False
            for method in METHODS:
                style = styles[method]
                if method != REFERENCE_METHOD:
                    dimensions, median, worst = _accuracy_series(
                        rows,
                        scenario,
                        experiment,
                        method,
                    )
                    if dimensions:
                        floor = 1e-16
                        accuracy_axis.plot(
                            dimensions,
                            [max(value, floor) for value in median],
                            linewidth=2.0,
                            markersize=5.0,
                            **style,
                        )
                        accuracy_axis.plot(
                            dimensions,
                            [max(value, floor) for value in worst],
                            linewidth=1.0,
                            linestyle="--",
                            alpha=0.55,
                            color=style["color"],
                        )

                dimensions, solved, capped, empty = _outcome_fractions(
                    rows,
                    scenario,
                    experiment,
                    method,
                )
                if not dimensions:
                    continue
                shift = offsets[method]
                line, = outcome_axis.plot(
                    dimensions,
                    [value + shift for value in solved],
                    linewidth=2.0,
                    markersize=5.0,
                    label=method,
                    **style,
                )
                outcome_axis.plot(
                    dimensions,
                    [value + shift for value in capped],
                    linewidth=1.0,
                    linestyle="--",
                    alpha=0.55,
                    color=style["color"],
                )
                if not empty_drawn and any(value > 0.0 for value in empty):
                    outcome_axis.fill_between(
                        dimensions,
                        0.0,
                        empty,
                        color="#999999",
                        alpha=0.22,
                        zorder=0,
                    )
                    empty_drawn = True
                if method not in labels:
                    handles.append(line)
                    labels.append(method)

    figure.suptitle(
        "Accuracy rows: relative objective difference from Gurobi on "
        "instances both solved (solid: median over seeds, dashed: worst "
        "seed); Gurobi is the reference and is not drawn there.\n"
        "Outcome rows: share of runs that solved to the requested "
        "tolerance (solid) or hit a time / iteration limit (dashed); grey "
        "band marks instances reported infeasible. Curves carry a small "
        "vertical offset so overlapping methods stay visible.",
        y=0.985,
        va="top",
        fontsize=12,
    )
    if handles:
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.925),
            ncol=5,
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
        "--gurobi-license-mode",
        choices=("auto", "default", "file"),
        default="file",
        help=(
            "Gurobi license selection: auto prefers the interpreter's default "
            "license and falls back to gurobi.lic; default forces the "
            "interpreter default; file forces gurobi.lic"
        ),
    )
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
        "--precision-plot",
        type=Path,
        default=DEFAULT_PRECISION_PLOT,
        help=(
            "companion figure: objective agreement with Gurobi, and the "
            "share of runs that solved, hit a limit, or were infeasible"
        ),
    )
    result.add_argument(
        "--exact-seed-count",
        type=int,
        default=DEFAULT_EXACT_SEED_COUNT,
        help=(
            "seeds per dimension for the exact sparse experiment, taken as "
            "a prefix of the shared seed list. Most exact rows cost the "
            "full time limit, so this is the main runtime control"
        ),
    )
    result.add_argument(
        "--exact-dimensions",
        type=int,
        nargs="+",
        default=list(DEFAULT_EXACT_DIMENSIONS),
        help=(
            "dimension ladder for the exact sparse experiment, intersected "
            "with --dimensions; overrides --exact-max-dimension. The "
            "default keeps full resolution up to n=1000 and two anchors "
            "above it, where every method saturates its time limit"
        ),
    )
    result.add_argument(
        "--exact-max-dimension",
        type=int,
        default=1000,
        help=(
            "skip the exact sparse experiment above this dimension. Every "
            "method is capped there anyway, so the rows cost the full time "
            "limit and report only the cap; prox and relaxation still run "
            "over the whole dimension ladder"
        ),
    )
    result.add_argument(
        "--resume",
        nargs="?",
        const=RESUME_FROM_OUTPUT,
        default=None,
        metavar="CHECKPOINT",
        help=(
            "reuse rows from an earlier checkpoint CSV instead of measuring "
            "them again; omitted (the default) recomputes every row, so one "
            "result file only ever holds rows measured in a single session. "
            "Bare --resume continues the file named by --output; pass a path "
            "to continue a different checkpoint"
        ),
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
    env_info = _bootstrap_commercial_environment(args.gurobi_license_mode)

    print("Four-scaling benchmark")
    print(f"Workspace root: {_workspace_root()}")
    print(f"Repository root: {_repository_root()}")
    print(f"Gurobi license: {env_info.get('gurobi_license')}")
    print(f"MOSEK license: {env_info.get('mosek_license')}")
    print(f"MOSEK site-packages: {env_info.get('mosek_site_packages')}")
    print(f"Dimensions: {tuple(args.dimensions)}")
    print(f"Seeds per dimension: {args.seed_count}")

    checkpoint, existing = _resume_rows(args.resume, args.output)
    if checkpoint is None:
        print("Resume: disabled; every row is measured in this run")
        _archive_previous_results(args.output)
    else:
        print(f"Resume checkpoint: {checkpoint}")
    keys_to_run: list[tuple[str, str, int, str, int]] = []
    for scenario, _ in SCENARIOS:
        for experiment in EXPERIMENTS:
            for dimension in _experiment_dimensions(
                experiment,
                args.dimensions,
                args.exact_max_dimension,
                args.exact_dimensions,
            ):
                seeds = _experiment_seeds(
                    experiment,
                    _seed_values(
                        args.seed,
                        scenario,
                        int(dimension),
                        args.seed_count,
                    ),
                    args.exact_seed_count,
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
                _experiment_dimensions(
                    experiment,
                    args.dimensions,
                    args.exact_max_dimension,
                    args.exact_dimensions,
                ),
                desc=f"{scenario}:{experiment}",
                leave=False,
                unit="n",
            )
            for dimension in experiment_bar:
                k = k_rule(int(dimension))
                seeds = _experiment_seeds(
                    experiment,
                    _seed_values(
                        args.seed,
                        scenario,
                        int(dimension),
                        args.seed_count,
                    ),
                    args.exact_seed_count,
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
    _precision_plot(args.precision_plot, all_rows)
    print(f"CSV: {args.output.resolve()}")
    print(f"Summary CSV: {summary_path.resolve()}")
    print(f"Plot: {args.plot.resolve()}")
    print(f"Precision plot: {args.precision_plot.resolve()}")
    print(f"Rows written this run: {len(fresh_rows)}")


if __name__ == "__main__":
    main()
