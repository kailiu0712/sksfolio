"""Run the continuous relaxation on Bertsimas--Cory-Wright-shaped cases."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np

from .bertsimas_cory_wright import (
    BCWCase,
    CONSTRAINT_PROFILES,
    HISTORICAL_UNIVERSES,
    generate_synthetic_case,
)
from ..relaxation import solve_relaxation
from ..relaxation.pdhg import PDHG_VARIANTS, PAVA_METHODS


DEFAULT_VARIANTS = PDHG_VARIANTS
DEFAULT_PAVA = PAVA_METHODS
DEFAULT_BACKENDS = ("fista", "pdhg", "gurobi.python")
SYNTHETIC_PROVENANCE = "deterministic_synthetic_factor_generator"
PDHG_CHECK_INTERVAL = 50
PDHG_MIN_EPOCH = 100
PDHG_MAX_EPOCH = 2000
FISTA_INITIAL_LIPSCHITZ = None
FISTA_BACKTRACKING_FACTOR = 2.0
FISTA_STEP_GROWTH = 1.1
FISTA_LINE_SEARCH_TOLERANCE = 1e-12
FISTA_ADAPTIVE_RESTART = True
FISTA_HISTORY_INTERVAL = 25
FISTA_PROX_TOLERANCE = 1e-8
FISTA_PROX_MAX_ITERATIONS = 1000
FISTA_MAX_BACKTRACKS = 60
FISTA_PROX_ORACLES = ("auto",)
FISTA_MAJOR_MAX_ITERATIONS = 20_000
REFERENCE_TOLERANCE = 1e-10
DEFAULT_CONSTRAINT_PROFILE = "bcw"


def _case(args: argparse.Namespace) -> BCWCase:
    _, processed, ranks = HISTORICAL_UNIVERSES[args.universe]
    if args.rank not in ranks:
        raise ValueError(
            f"rank must be one of {ranks} for {args.universe}"
        )
    return BCWCase(
        family="historical",
        universe=args.universe,
        dimension=processed,
        rank=args.rank,
        k=args.k,
        gamma_scale=args.gamma_scale,
        regime=args.regime,
    )


def _row(
    case: BCWCase,
    problem: Any,
    backend: str,
    variant: str,
    pava: str,
    result: Any,
    run_configuration: Dict[str, Any],
) -> Dict[str, Any]:
    diagnostics = result.raw.get("diagnostics", {})
    certificate = result.dual_certificate
    violations = diagnostics.get("violations", {})
    maximum_violation = violations.get("maximum")
    constraint_names = tuple(
        getattr(problem, "constraint_names", ())
    )
    lower = np.asarray(getattr(problem, "lower", ()), dtype=float)
    upper = np.asarray(getattr(problem, "upper", ()), dtype=float)
    finite_lower = np.isfinite(lower)
    finite_upper = np.isfinite(upper)
    equalities = finite_lower & finite_upper & (lower == upper)
    two_sided = finite_lower & finite_upper & ~equalities
    lower_only = finite_lower & ~finite_upper
    upper_only = ~finite_lower & finite_upper
    constraint_signature = hashlib.sha256(
        "\n".join(constraint_names).encode("utf-8")
    ).hexdigest()[:12]
    profile = run_configuration["constraint_profile"]
    profile_suffix = "" if profile == "bcw" else f"-{profile}"
    return {
        "case": case.key,
        "instance_id": (
            f"{case.key}-{run_configuration['instance_provenance']}"
            f"-seed{run_configuration['seed']}{profile_suffix}"
        ),
        "family": case.family,
        "universe": case.universe,
        "dimension": case.dimension,
        "rank": case.rank,
        "k": case.k,
        "gamma_scale": case.gamma_scale,
        "gamma": case.gamma,
        "regime": case.regime,
        "constraint_count": len(constraint_names),
        "constraint_names": "|".join(constraint_names),
        "constraint_signature": constraint_signature,
        "constraint_nnz": (
            int(problem.C.nnz)
            if hasattr(getattr(problem, "C", None), "nnz")
            else int(np.count_nonzero(getattr(problem, "C", ())))
        ),
        "equality_row_count": int(np.count_nonzero(equalities)),
        "two_sided_row_count": int(np.count_nonzero(two_sided)),
        "lower_only_row_count": int(np.count_nonzero(lower_only)),
        "upper_only_row_count": int(np.count_nonzero(upper_only)),
        **run_configuration,
        "backend": backend,
        "variant": variant,
        "pava": pava,
        "prox_oracle": result.raw.get("prox_oracle_used"),
        "status": result.status,
        "objective": result.objective,
        "safe_dual_bound": result.safe_dual_bound,
        "certificate_verified": (
            certificate.verify(problem)
            if certificate is not None
            else None
        ),
        "anchor_safe_gap": result.raw.get(
            "anchor_primal_dual_gap"
        ),
        "iterations": result.raw.get("iterations"),
        "residual": result.raw.get("residual"),
        "relative_residual": result.raw.get("relative_residual"),
        "line_search_backtracks": result.raw.get(
            "line_search_backtracks"
        ),
        "restarts": result.raw.get("restarts"),
        "gradient_evaluations": result.raw.get(
            "gradient_evaluations"
        ),
        "prox_calls": result.raw.get("prox_calls"),
        "pava_calls": result.raw.get("pava_calls"),
        "prox_scalar_evaluations": result.raw.get(
            "prox_scalar_evaluations"
        ),
        "dual_bound_evaluations": result.raw.get(
            "dual_bound_evaluations"
        ),
        "initial_lipschitz": result.raw.get("initial_lipschitz"),
        "final_lipschitz": result.raw.get("final_lipschitz"),
        "budget_value": diagnostics.get("budget"),
        "linear_row_violation": violations.get("linear_rows"),
        "nonnegativity_violation": violations.get(
            "nonnegativity"
        ),
        "upper_box_violation": violations.get("upper_box"),
        "perspective_budget_violation": violations.get(
            "perspective_budget"
        ),
        "maximum_violation": maximum_violation,
        "primal_feasible": (
            maximum_violation
            <= run_configuration["feasibility_tolerance"]
            if maximum_violation is not None
            else None
        ),
        "solve_seconds": result.raw.get("solve_seconds"),
        "total_seconds": result.raw.get("total_seconds"),
        "wrapper_seconds": result.raw.get("wrapper_seconds"),
    }


def _optimal_gurobi_reference(
    rows: Iterable[Dict[str, Any]],
) -> Any:
    """Return a native-Gurobi optimum, never a time-limited incumbent."""
    return next(
        (
            row["objective"]
            for row in rows
            if row["backend"] == "gurobi.python"
            and row["status"] == "optimal"
            and row["objective"] is not None
            and row.get("primal_feasible") is True
        ),
        None,
    )


def _run_configuration(
    *,
    seed: int,
    target_iterations: int,
    tolerance: float,
    max_iterations: int,
    time_limit: float,
    threads: int,
    fista_initial_lipschitz: float | None = FISTA_INITIAL_LIPSCHITZ,
    fista_backtracking_factor: float = FISTA_BACKTRACKING_FACTOR,
    fista_step_growth: float = FISTA_STEP_GROWTH,
    fista_line_search_tolerance: float = FISTA_LINE_SEARCH_TOLERANCE,
    fista_adaptive_restart: bool = FISTA_ADAPTIVE_RESTART,
    fista_history_interval: int = FISTA_HISTORY_INTERVAL,
    fista_prox_tolerance: float = FISTA_PROX_TOLERANCE,
    fista_prox_max_iterations: int = FISTA_PROX_MAX_ITERATIONS,
    fista_max_backtracks: int = FISTA_MAX_BACKTRACKS,
    fista_prox_oracles: Iterable[str] = FISTA_PROX_ORACLES,
    fista_major_max_iterations: int = FISTA_MAJOR_MAX_ITERATIONS,
    reference_tolerance: float = REFERENCE_TOLERANCE,
    constraint_profile: str = DEFAULT_CONSTRAINT_PROFILE,
    constraint_metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return every non-case option needed to interpret a result row."""
    return {
        "instance_provenance": SYNTHETIC_PROVENANCE,
        "paper_returns_used": False,
        "seed": seed,
        "target_iterations": target_iterations,
        "tolerance": tolerance,
        "feasibility_tolerance": tolerance,
        "max_iterations": max_iterations,
        "time_limit": time_limit,
        "threads": threads,
        "commercial_log": False,
        "reference_tolerance": reference_tolerance,
        "effective_commercial_tolerance": min(
            tolerance,
            reference_tolerance,
        ),
        "constraint_profile": constraint_profile,
        "sector_count": (
            constraint_metadata or {}
        ).get("sectors"),
        "style_factor_count": (
            constraint_metadata or {}
        ).get("style_factors"),
        "stress_constraint_count": (
            constraint_metadata or {}
        ).get("stress_constraints"),
        "sector_band": (
            constraint_metadata or {}
        ).get("sector_band"),
        "style_band": (
            constraint_metadata or {}
        ).get("style_band"),
        "stress_band": (
            constraint_metadata or {}
        ).get("stress_band"),
        "pdhg_check_interval": PDHG_CHECK_INTERVAL,
        "pdhg_min_epoch": PDHG_MIN_EPOCH,
        "pdhg_max_epoch": PDHG_MAX_EPOCH,
        "fista_initial_lipschitz": fista_initial_lipschitz,
        "fista_backtracking_factor": fista_backtracking_factor,
        "fista_step_growth": fista_step_growth,
        "fista_line_search_tolerance": (
            fista_line_search_tolerance
        ),
        "fista_adaptive_restart": fista_adaptive_restart,
        "fista_history_interval": fista_history_interval,
        "fista_prox_tolerance": fista_prox_tolerance,
        "fista_prox_max_iterations": fista_prox_max_iterations,
        "fista_max_backtracks": fista_max_backtracks,
        "fista_prox_oracles": "|".join(fista_prox_oracles),
        "fista_major_max_iterations": fista_major_max_iterations,
    }


def run(
    case: BCWCase,
    *,
    backends: Iterable[str],
    variants: Iterable[str],
    pava_methods: Iterable[str],
    seed: int,
    target_iterations: int,
    tolerance: float,
    max_iterations: int,
    time_limit: float,
    threads: int,
    fista_initial_lipschitz: float | None = FISTA_INITIAL_LIPSCHITZ,
    fista_backtracking_factor: float = FISTA_BACKTRACKING_FACTOR,
    fista_step_growth: float = FISTA_STEP_GROWTH,
    fista_line_search_tolerance: float = FISTA_LINE_SEARCH_TOLERANCE,
    fista_adaptive_restart: bool = FISTA_ADAPTIVE_RESTART,
    fista_history_interval: int = FISTA_HISTORY_INTERVAL,
    fista_prox_tolerance: float = FISTA_PROX_TOLERANCE,
    fista_prox_max_iterations: int = FISTA_PROX_MAX_ITERATIONS,
    fista_max_backtracks: int = FISTA_MAX_BACKTRACKS,
    fista_prox_oracles: Iterable[str] = FISTA_PROX_ORACLES,
    fista_major_max_iterations: int = FISTA_MAJOR_MAX_ITERATIONS,
    reference_tolerance: float = REFERENCE_TOLERANCE,
    constraint_profile: str = DEFAULT_CONSTRAINT_PROFILE,
    sectors: int | None = None,
    style_factors: int | None = None,
    stress_constraints: int | None = None,
    sector_band: float | None = None,
    style_band: float | None = None,
    stress_band: float | None = None,
) -> list[Dict[str, Any]]:
    """Run all requested configurations and return flat result rows."""
    fista_prox_oracles = tuple(fista_prox_oracles)
    problem = generate_synthetic_case(
        case,
        seed=seed,
        target_iterations=target_iterations,
        constraint_profile=constraint_profile,
        sectors=sectors,
        style_factors=style_factors,
        stress_constraints=stress_constraints,
        sector_band=sector_band,
        style_band=style_band,
        stress_band=stress_band,
    )
    run_configuration = _run_configuration(
        seed=seed,
        target_iterations=target_iterations,
        tolerance=tolerance,
        max_iterations=max_iterations,
        time_limit=time_limit,
        threads=threads,
        fista_initial_lipschitz=fista_initial_lipschitz,
        fista_backtracking_factor=fista_backtracking_factor,
        fista_step_growth=fista_step_growth,
        fista_line_search_tolerance=fista_line_search_tolerance,
        fista_adaptive_restart=fista_adaptive_restart,
        fista_history_interval=fista_history_interval,
        fista_prox_tolerance=fista_prox_tolerance,
        fista_prox_max_iterations=fista_prox_max_iterations,
        fista_max_backtracks=fista_max_backtracks,
        fista_prox_oracles=fista_prox_oracles,
        fista_major_max_iterations=fista_major_max_iterations,
        reference_tolerance=reference_tolerance,
        constraint_profile=constraint_profile,
        constraint_metadata=getattr(problem, "metadata", {}).get(
            "constraints",
            {},
        ),
    )
    backends = tuple(backends)
    variants = tuple(variants)
    pava_methods = tuple(pava_methods)
    rows = []
    for backend in backends:
        if backend == "pdhg":
            for variant in variants:
                for pava in pava_methods:
                    result = solve_relaxation(
                        problem,
                        backend,
                        variant=variant,
                        pava=pava,
                        options={
                            "tolerance": tolerance,
                            "feasibility_tolerance": tolerance,
                            "max_iterations": max_iterations,
                            "time_limit": time_limit,
                            "threads": threads,
                            "check_interval": PDHG_CHECK_INTERVAL,
                            "min_epoch": PDHG_MIN_EPOCH,
                            "max_epoch": PDHG_MAX_EPOCH,
                        },
                    )
                    rows.append(
                        _row(
                            case,
                            problem,
                            backend,
                            variant,
                            pava,
                            result,
                            run_configuration,
                        )
                    )
        elif backend == "fista":
            constraint_names = tuple(problem.constraint_names)
            for requested_oracle in fista_prox_oracles:
                if requested_oracle == "pava" and constraint_names:
                    continue
                if (
                    requested_oracle == "budget"
                    and constraint_names != ("budget",)
                ):
                    continue
                oracle_pava_methods = (
                    ("",)
                    if requested_oracle == "majorization_qp"
                    else pava_methods
                )
                for pava in oracle_pava_methods:
                    result = solve_relaxation(
                        problem,
                        backend,
                        pava=(
                            "partial_sort" if not pava else pava
                        ),
                        options={
                            "tolerance": tolerance,
                            "feasibility_tolerance": tolerance,
                            "max_iterations": max_iterations,
                            "time_limit": time_limit,
                            "threads": threads,
                            "initial_lipschitz": (
                                fista_initial_lipschitz
                            ),
                            "backtracking_factor": (
                                fista_backtracking_factor
                            ),
                            "step_growth": fista_step_growth,
                            "line_search_tolerance": (
                                fista_line_search_tolerance
                            ),
                            "adaptive_restart": (
                                fista_adaptive_restart
                            ),
                            "history_interval": (
                                fista_history_interval
                            ),
                            "prox_tolerance": fista_prox_tolerance,
                            "prox_max_iterations": (
                                fista_prox_max_iterations
                            ),
                            "prox_oracle": requested_oracle,
                            "majorization_max_iterations": (
                                fista_major_max_iterations
                            ),
                            "max_backtracks": fista_max_backtracks,
                        },
                    )
                    rows.append(
                        _row(
                            case,
                            problem,
                            backend,
                            str(result.raw.get("variant", "")),
                            pava,
                            result,
                            run_configuration,
                        )
                    )
        else:
            commercial_tolerance = run_configuration[
                "effective_commercial_tolerance"
            ]
            result = solve_relaxation(
                problem,
                backend,
                options={
                    "tolerance": commercial_tolerance,
                    "time_limit": time_limit,
                    "threads": threads,
                    "log": run_configuration["commercial_log"],
                },
            )
            rows.append(
                _row(
                    case,
                    problem,
                    backend,
                    "",
                    "",
                    result,
                    run_configuration,
                )
            )
    reference_objective = _optimal_gurobi_reference(rows)
    for row in rows:
        row["reference_backend"] = (
            "gurobi.python" if reference_objective is not None else None
        )
        row["reference_objective"] = reference_objective
        bound = row["safe_dual_bound"]
        row["safe_dual_gap_to_reference"] = (
            reference_objective - bound
            if reference_objective is not None and bound is not None
            else None
        )
        safe_gap = row["safe_dual_gap_to_reference"]
        row["safe_dual_relative_gap_to_reference"] = (
            safe_gap / max(1.0, abs(reference_objective))
            if safe_gap is not None and reference_objective is not None
            else None
        )
        objective = row["objective"]
        row["objective_error_to_reference"] = (
            objective - reference_objective
            if reference_objective is not None
            and objective is not None
            else None
        )
        row["feasible_objective_gap_to_reference"] = (
            objective - reference_objective
            if reference_objective is not None
            and objective is not None
            and row["primal_feasible"] is True
            else None
        )
    return rows


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        description=(
            "Compare first-order methods with commercial solvers "
            "on a synthetic, paper-shaped continuous relaxation"
        )
    )
    argument_parser.add_argument(
        "--output",
        type=Path,
        default=Path("bcw_relaxation_results.csv"),
    )
    argument_parser.add_argument(
        "--universe",
        choices=tuple(HISTORICAL_UNIVERSES),
        default="sp500",
    )
    argument_parser.add_argument("--rank", type=int, default=50)
    argument_parser.add_argument("--k", type=int, default=10)
    argument_parser.add_argument(
        "--gamma-scale",
        type=float,
        choices=(1.0, 100.0),
        default=100.0,
    )
    argument_parser.add_argument(
        "--regime",
        choices=("unconstrained", "constrained"),
        default="unconstrained",
    )
    argument_parser.add_argument(
        "--constraint-profile",
        choices=tuple(CONSTRAINT_PROFILES),
        default=DEFAULT_CONSTRAINT_PROFILE,
        help=(
            "bcw keeps only the paper rows; standard adds the earlier "
            "20-sector, 8-style, 15-stress stack; many adds 20 sector, "
            "40 style, and 200 stress rows"
        ),
    )
    argument_parser.add_argument(
        "--sectors",
        type=int,
        default=None,
        help="override the selected profile's number of sector bands",
    )
    argument_parser.add_argument(
        "--style-factors",
        type=int,
        default=None,
        help="override the selected profile's number of style-factor bands",
    )
    argument_parser.add_argument(
        "--stress-constraints",
        type=int,
        default=None,
        help="override the selected profile's number of stress-loss rows",
    )
    argument_parser.add_argument(
        "--sector-band",
        type=float,
        default=None,
    )
    argument_parser.add_argument(
        "--style-band",
        type=float,
        default=None,
    )
    argument_parser.add_argument(
        "--stress-band",
        type=float,
        default=None,
    )
    argument_parser.add_argument(
        "--backend",
        action="append",
        dest="backends",
        choices=(
            "pdhg",
            "fista",
            "gurobi.python",
            "gurobi.julia",
            "mosek.python",
            "mosek.julia",
        ),
        help=(
            "repeat to select backends; defaults depend on the entry point"
        ),
    )
    argument_parser.add_argument(
        "--variant",
        action="append",
        dest="variants",
        choices=PDHG_VARIANTS,
        help="repeat to select PDHG variants; default is all",
    )
    argument_parser.add_argument(
        "--pava",
        action="append",
        dest="pava_methods",
        choices=PAVA_METHODS,
        help="repeat to select PAVA methods; default is both",
    )
    argument_parser.add_argument("--seed", type=int, default=7)
    argument_parser.add_argument(
        "--target-iterations",
        type=int,
        default=200,
    )
    argument_parser.add_argument("--tolerance", type=float, default=1e-6)
    argument_parser.add_argument(
        "--max-iterations",
        type=int,
        default=100000,
    )
    argument_parser.add_argument("--time-limit", type=float, default=600.0)
    argument_parser.add_argument("--threads", type=int, default=1)
    argument_parser.add_argument(
        "--reference-tolerance",
        type=float,
        default=REFERENCE_TOLERANCE,
        help=(
            "commercial-reference tolerance; the tighter of this "
            "and --tolerance is used"
        ),
    )
    argument_parser.add_argument(
        "--fista-initial-lipschitz",
        type=float,
        default=FISTA_INITIAL_LIPSCHITZ,
        help="initial FISTA line-search estimate; default is automatic",
    )
    argument_parser.add_argument(
        "--fista-backtracking-factor",
        type=float,
        default=FISTA_BACKTRACKING_FACTOR,
    )
    argument_parser.add_argument(
        "--fista-step-growth",
        type=float,
        default=FISTA_STEP_GROWTH,
    )
    argument_parser.add_argument(
        "--fista-line-search-tolerance",
        type=float,
        default=FISTA_LINE_SEARCH_TOLERANCE,
    )
    argument_parser.add_argument(
        "--fista-adaptive-restart",
        action=argparse.BooleanOptionalAction,
        default=FISTA_ADAPTIVE_RESTART,
    )
    argument_parser.add_argument(
        "--fista-history-interval",
        type=int,
        default=FISTA_HISTORY_INTERVAL,
    )
    argument_parser.add_argument(
        "--fista-prox-tolerance",
        type=float,
        default=FISTA_PROX_TOLERANCE,
    )
    argument_parser.add_argument(
        "--fista-prox-max-iterations",
        type=int,
        default=FISTA_PROX_MAX_ITERATIONS,
    )
    argument_parser.add_argument(
        "--fista-prox-oracle",
        action="append",
        dest="fista_prox_oracles",
        choices=(
            "auto",
            "pava",
            "budget",
            "dual_fista",
            "majorization_qp",
        ),
        help=(
            "repeat to compare FISTA prox oracles; default is automatic"
        ),
    )
    argument_parser.add_argument(
        "--fista-major-max-iterations",
        type=int,
        default=FISTA_MAJOR_MAX_ITERATIONS,
    )
    argument_parser.add_argument(
        "--fista-max-backtracks",
        type=int,
        default=FISTA_MAX_BACKTRACKS,
    )
    return argument_parser


def _main(
    arguments: Sequence[str] | None,
    *,
    default_backends: Sequence[str],
    default_output: Path,
    default_tolerance: float,
    budget_only: bool,
    default_variants: Sequence[str] = DEFAULT_VARIANTS,
    default_pava: Sequence[str] = DEFAULT_PAVA,
    default_fista_prox_oracles: Sequence[str] = FISTA_PROX_ORACLES,
    default_constraint_profile: str = DEFAULT_CONSTRAINT_PROFILE,
    default_regime: str = "unconstrained",
) -> None:
    argument_parser = parser()
    argument_parser.set_defaults(
        output=default_output,
        tolerance=default_tolerance,
        constraint_profile=default_constraint_profile,
        regime=default_regime,
    )
    args = argument_parser.parse_args(arguments)
    if budget_only and args.regime != "unconstrained":
        argument_parser.error(
            "the matched PDHG--FISTA benchmark requires "
            "--regime unconstrained"
        )
    case = _case(args)
    rows = run(
        case,
        backends=args.backends or default_backends,
        variants=args.variants or default_variants,
        pava_methods=args.pava_methods or default_pava,
        seed=args.seed,
        target_iterations=args.target_iterations,
        tolerance=args.tolerance,
        max_iterations=args.max_iterations,
        time_limit=args.time_limit,
        threads=args.threads,
        fista_initial_lipschitz=args.fista_initial_lipschitz,
        fista_backtracking_factor=args.fista_backtracking_factor,
        fista_step_growth=args.fista_step_growth,
        fista_line_search_tolerance=(
            args.fista_line_search_tolerance
        ),
        fista_adaptive_restart=args.fista_adaptive_restart,
        fista_history_interval=args.fista_history_interval,
        fista_prox_tolerance=args.fista_prox_tolerance,
        fista_prox_max_iterations=args.fista_prox_max_iterations,
        fista_prox_oracles=(
            args.fista_prox_oracles or default_fista_prox_oracles
        ),
        fista_major_max_iterations=(
            args.fista_major_max_iterations
        ),
        fista_max_backtracks=args.fista_max_backtracks,
        reference_tolerance=args.reference_tolerance,
        constraint_profile=args.constraint_profile,
        sectors=args.sectors,
        style_factors=args.style_factors,
        stress_constraints=args.stress_constraints,
        sector_band=args.sector_band,
        style_band=args.style_band,
        stress_band=args.stress_band,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(args.output.resolve())


def main(arguments: Sequence[str] | None = None) -> None:
    _main(
        arguments,
        default_backends=DEFAULT_BACKENDS,
        default_output=Path("bcw_relaxation_results.csv"),
        default_tolerance=1e-6,
        budget_only=False,
    )


if __name__ == "__main__":
    main()
