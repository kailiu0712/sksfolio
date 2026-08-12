"""Paper-calibrated and synthetic large Markowitz instance generation.

The historical recipe follows Bertsimas and Cory-Wright (2021):
daily returns, one-month rescaling, a truncated SVD of the correlation
matrix, per-asset volatility rescaling, and their 30% return target.

The paper does not prescribe a random generator. The synthetic mode is
therefore explicitly labeled as an adaptation that preserves the paper's
factor ranks, k values, gamma grid, and monthly scaling.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds

from ..relaxation.problem import MarkowitzInstance, save_instance_bundle


PAPER_URL = "https://arxiv.org/abs/1811.00138"
DEFAULT_TRADING_DAYS_PER_MONTH = 2769.0 / (11.0 * 12.0)


def project_simplex(values: Any) -> np.ndarray:
    """Euclidean projection onto {x >= 0, 1' x = 1}."""
    vector = np.asarray(values, dtype=float).reshape(-1)
    ordered = np.sort(vector)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    indices = np.arange(1, vector.size + 1)
    active = ordered - cumulative / indices > 0.0
    if not np.any(active):
        return np.full(vector.size, 1.0 / vector.size)
    rho = int(np.nonzero(active)[0][-1])
    threshold = cumulative[rho] / float(rho + 1)
    result = np.maximum(vector - threshold, 0.0)
    return result / float(np.sum(result))


def factor_operator_norm_squared(
    factor_loadings: np.ndarray,
    iterations: int = 12,
    seed: int = 941,
) -> float:
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=factor_loadings.shape[1])
    vector /= max(float(np.linalg.norm(vector)), 1e-16)
    eigenvalue = 0.0
    for _ in range(max(2, iterations)):
        image = factor_loadings.T @ (factor_loadings @ vector)
        norm = float(np.linalg.norm(image))
        if norm <= 1e-30:
            return 0.0
        vector = image / norm
        eigenvalue = float(
            vector @ (factor_loadings.T @ (factor_loadings @ vector))
        )
    return max(eigenvalue, 0.0)


def paper_return_target(
    factor_loadings: np.ndarray,
    expected_returns: np.ndarray,
    gamma: float,
    fraction: float = 0.3,
    max_iterations: int = 200,
    tolerance: float = 1e-9,
) -> Tuple[float, np.ndarray, Dict[str, Any]]:
    """Compute the paper's r_min + fraction * (r_max - r_min) target.

    The x_max problem is solved exactly by simplex projection. The x_min
    factor QP is solved by accelerated projected gradient without ever
    forming the dense covariance matrix.
    """
    if gamma <= 0.0 or not math.isfinite(gamma):
        raise ValueError("gamma must be positive and finite")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must lie in [0, 1]")
    dimension = factor_loadings.shape[0]
    x = np.full(dimension, 1.0 / dimension)
    y = x.copy()
    momentum = 1.0
    norm_squared = factor_operator_norm_squared(factor_loadings)
    lipschitz = norm_squared + 1.0 / gamma
    completed = 0
    relative_change = math.inf
    for iteration in range(1, max_iterations + 1):
        gradient = (
            factor_loadings @ (factor_loadings.T @ y)
            + y / gamma
        )
        candidate = project_simplex(y - gradient / lipschitz)
        relative_change = float(
            np.linalg.norm(candidate - x)
            / max(1.0, np.linalg.norm(x))
        )
        next_momentum = 0.5 * (
            1.0 + math.sqrt(1.0 + 4.0 * momentum * momentum)
        )
        y = candidate + (
            (momentum - 1.0) / next_momentum
        ) * (candidate - x)
        x = candidate
        momentum = next_momentum
        completed = iteration
        if relative_change <= tolerance:
            break

    x_min = x
    x_max = project_simplex(gamma * expected_returns)
    r_min = float(expected_returns @ x_min)
    r_max = float(expected_returns @ x_max)
    if r_max < r_min:
        r_min, r_max = r_max, r_min
        x_min, x_max = x_max, x_min
    target = r_min + fraction * (r_max - r_min)
    anchor = (1.0 - fraction) * x_min + fraction * x_max
    anchor /= float(np.sum(anchor))
    return target, anchor, {
        "r_min": r_min,
        "r_max": r_max,
        "target_fraction": fraction,
        "target": target,
        "x_min_iterations": completed,
        "x_min_relative_change": relative_change,
        "factor_operator_norm_squared": norm_squared,
    }


def synthetic_factor_data(
    dimension: int,
    rank: int,
    seed: int,
    annual_volatility: float = 0.20,
    common_correlation: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generate a monthly low-rank covariance in the paper's format."""
    if dimension < 10:
        raise ValueError("dimension must be at least 10")
    if not 2 <= rank < dimension:
        raise ValueError("rank must lie in {2, ..., dimension - 1}")
    if not 0.0 <= common_correlation < 1.0:
        raise ValueError("common_correlation must lie in [0, 1)")
    rng = np.random.default_rng(seed)
    loadings = rng.normal(
        scale=math.sqrt((1.0 - common_correlation) / (rank - 1)),
        size=(dimension, rank),
    )
    loadings[:, 0] = math.sqrt(common_correlation)
    row_norms = np.linalg.norm(loadings, axis=1)
    loadings /= np.maximum(row_norms[:, None], 1e-16)
    monthly_volatility = np.exp(
        rng.normal(
            loc=math.log(annual_volatility / math.sqrt(12.0)),
            scale=0.25,
            size=dimension,
        )
    )
    monthly_volatility = np.clip(
        monthly_volatility,
        0.08 / math.sqrt(12.0),
        0.60 / math.sqrt(12.0),
    )
    factor_loadings = monthly_volatility[:, None] * loadings

    factor_premia = rng.normal(size=rank)
    factor_premia /= max(float(np.linalg.norm(factor_premia)), 1e-16)
    return_score = (
        loadings @ factor_premia
        + 0.35 * rng.normal(size=dimension)
    )
    order = np.argsort(np.argsort(return_score)).astype(float)
    order /= max(1.0, float(dimension - 1))
    annual_return = 0.03 + 0.15 * order
    expected_returns = np.power(1.0 + annual_return, 1.0 / 12.0) - 1.0
    return factor_loadings, expected_returns, {
        "mode": "synthetic-factor-adaptation",
        "annual_volatility_center": annual_volatility,
        "common_correlation": common_correlation,
    }


def historical_factor_data(
    daily_returns: Any,
    rank: int,
    outlier_cutoff: float = 0.20,
    outlier_mode: str = "cell",
    trading_days_per_month: float = DEFAULT_TRADING_DAYS_PER_MONTH,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Apply the paper's low-rank monthly covariance recipe.

    The paper does not clarify whether a >20% observation removes one
    asset observation or the entire day. ``cell`` treats it as an
    asset-level missing observation and mean-imputes after centering;
    ``row`` removes the entire day.
    """
    returns = np.asarray(daily_returns, dtype=float)
    if returns.ndim != 2:
        raise ValueError("daily_returns must be a samples-by-assets matrix")
    if returns.shape[0] < rank + 2 or returns.shape[1] <= rank:
        raise ValueError("rank is too large for the returns matrix")
    if outlier_mode not in {"cell", "row", "none"}:
        raise ValueError("outlier_mode must be cell, row, or none")
    cleaned = returns.copy()
    if outlier_mode == "row":
        cleaned = cleaned[
            np.all(np.abs(cleaned) <= outlier_cutoff, axis=1)
        ]
    elif outlier_mode == "cell":
        cleaned[np.abs(cleaned) > outlier_cutoff] = np.nan
    means = np.nanmean(cleaned, axis=0)
    centered = cleaned - means
    deviations = np.nanstd(cleaned, axis=0, ddof=1)
    if np.any(~np.isfinite(deviations)) or np.any(deviations <= 0.0):
        raise ValueError("every asset must have positive return variance")
    standardized = centered / deviations
    standardized = np.where(np.isfinite(standardized), standardized, 0.0)
    _, singular_values, right_vectors = svds(
        standardized,
        k=rank,
        return_singular_vectors=True,
        which="LM",
    )
    order = np.argsort(singular_values)[::-1]
    singular_values = singular_values[order]
    right_vectors = right_vectors[order, :]
    daily_factor = (
        deviations[:, None]
        * right_vectors.T
        * (singular_values / math.sqrt(cleaned.shape[0] - 1.0))
    )
    factor_loadings = math.sqrt(trading_days_per_month) * daily_factor
    expected_returns = trading_days_per_month * means
    return factor_loadings, expected_returns, {
        "mode": "historical-paper-recipe",
        "observations_before_filter": int(returns.shape[0]),
        "observations_after_filter": int(cleaned.shape[0]),
        "outlier_cutoff": outlier_cutoff,
        "outlier_mode": outlier_mode,
        "trading_days_per_month": trading_days_per_month,
    }


def build_constraint_matrix(
    factor_loadings: np.ndarray,
    expected_returns: np.ndarray,
    anchor: np.ndarray,
    target_return: Optional[float],
    sectors: int,
    style_factors: int,
    stress_constraints: int,
    sector_band: float,
    style_band: float,
    stress_band: float,
    seed: int,
) -> Tuple[sparse.csc_matrix, np.ndarray, np.ndarray, list[str]]:
    dimension = expected_returns.size
    counts = {
        "sectors": sectors,
        "style_factors": style_factors,
        "stress_constraints": stress_constraints,
    }
    for name, value in counts.items():
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < 0
        ):
            raise ValueError(f"{name} must be a nonnegative integer")
    if sectors > dimension:
        raise ValueError("sectors cannot exceed the number of assets")
    for name, value in (
        ("sector_band", sector_band),
        ("style_band", style_band),
        ("stress_band", stress_band),
    ):
        if not math.isfinite(float(value)) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    rng = np.random.default_rng(seed + 711)
    row_parts = []
    column_parts = []
    value_parts = []
    lower = []
    upper = []
    names = []
    columns = np.arange(dimension, dtype=np.int64)
    row = 0

    def add_dense(
        values: np.ndarray,
        row_lower: float,
        row_upper: float,
        name: str,
    ) -> None:
        nonlocal row
        row_parts.append(np.full(dimension, row, dtype=np.int64))
        column_parts.append(columns)
        value_parts.append(np.asarray(values, dtype=float))
        lower.append(row_lower)
        upper.append(row_upper)
        names.append(name)
        row += 1

    add_dense(np.ones(dimension), 1.0, 1.0, "budget")
    if target_return is not None:
        add_dense(
            expected_returns,
            float(target_return),
            math.inf,
            "minimum_return",
        )

    if sectors > 0:
        permutation = rng.permutation(dimension)
        sector_ids = np.empty(dimension, dtype=np.int64)
        sector_ids[permutation] = np.arange(dimension) % sectors
        row_parts.append(row + sector_ids)
        column_parts.append(columns)
        value_parts.append(np.ones(dimension))
        for sector in range(sectors):
            exposure = float(np.sum(anchor[sector_ids == sector]))
            lower.append(max(0.0, exposure - sector_band))
            upper.append(min(1.0, exposure + sector_band))
            names.append(f"sector_{sector}")
        row += sectors

    for style in range(style_factors):
        values = rng.normal(size=dimension)
        values -= np.mean(values)
        values /= max(float(np.std(values)), 1e-12)
        exposure = float(values @ anchor)
        add_dense(
            values,
            exposure - style_band,
            exposure + style_band,
            f"style_{style}",
        )

    for stress in range(stress_constraints):
        direction = rng.normal(size=factor_loadings.shape[1])
        direction /= max(float(np.linalg.norm(direction)), 1e-16)
        losses = (
            0.04
            - factor_loadings @ direction
            + 0.01 * rng.normal(size=dimension)
        )
        exposure = float(losses @ anchor)
        add_dense(
            losses,
            -math.inf,
            exposure + stress_band,
            f"stress_loss_{stress}",
        )

    row_indices = np.concatenate(row_parts)
    column_indices = np.concatenate(column_parts)
    data = np.concatenate(value_parts)
    matrix = sparse.coo_matrix(
        (data, (row_indices, column_indices)),
        shape=(row, dimension),
    ).tocsc()
    matrix.sum_duplicates()
    matrix.sort_indices()
    return (
        matrix,
        np.asarray(lower, dtype=float),
        np.asarray(upper, dtype=float),
        names,
    )


def generate_instance(
    dimension: int,
    rank: int,
    k: int,
    gamma_scale: float,
    regime: str,
    seed: int,
    sectors: int,
    style_factors: int,
    stress_constraints: int,
    target_fraction: float,
    target_iterations: int,
    sector_band: float,
    style_band: float,
    stress_band: float,
    annual_volatility: float,
    common_correlation: float,
    daily_returns: Optional[np.ndarray] = None,
    outlier_mode: str = "cell",
) -> MarkowitzInstance:
    if gamma_scale <= 0.0:
        raise ValueError("gamma_scale must be positive")
    if regime not in {"unconstrained", "constrained", "hybrid"}:
        raise ValueError("unknown regime")
    if daily_returns is None:
        if not 1 <= k <= dimension:
            raise ValueError("k must lie in {1, ..., dimension}")
        factor_loadings, expected_returns, data_metadata = (
            synthetic_factor_data(
                dimension,
                rank,
                seed,
                annual_volatility=annual_volatility,
                common_correlation=common_correlation,
            )
        )
    else:
        factor_loadings, expected_returns, data_metadata = (
            historical_factor_data(
                daily_returns,
                rank,
                outlier_mode=outlier_mode,
            )
        )
        dimension = int(expected_returns.size)
        if not 1 <= k <= dimension:
            raise ValueError("k exceeds the historical universe size")

    gamma = gamma_scale / math.sqrt(dimension)
    target, target_anchor, target_metadata = paper_return_target(
        factor_loadings,
        expected_returns,
        gamma,
        fraction=target_fraction,
        max_iterations=target_iterations,
    )
    target_row = target if regime in {"constrained", "hybrid"} else None
    return_reward = 1.0 if regime in {"unconstrained", "hybrid"} else 0.0
    constraint_matrix, lower, upper, names = build_constraint_matrix(
        factor_loadings,
        expected_returns,
        target_anchor,
        target_row,
        sectors,
        style_factors,
        stress_constraints,
        sector_band,
        style_band,
        stress_band,
        seed,
    )
    metadata = {
        "paper": PAPER_URL,
        "paper_model": (
            "continuous perspective relaxation of equation (34)"
        ),
        "dimension": dimension,
        "rank": rank,
        "seed": seed,
        "regime": regime,
        "gamma": gamma,
        "gamma_scale": gamma_scale,
        "perspective_weight": 1.0 / gamma,
        "k": k,
        "data": data_metadata,
        "return_target": target_metadata,
        "constraints": {
            "sectors": sectors,
            "style_factors": style_factors,
            "stress_constraints": stress_constraints,
            "sector_band": sector_band,
            "style_band": style_band,
            "stress_band": stress_band,
        },
    }
    instance = MarkowitzInstance(
        factor_loadings=factor_loadings,
        expected_returns=expected_returns,
        constraint_matrix=constraint_matrix,
        lower_bounds=lower,
        upper_bounds=upper,
        feasible_anchor=target_anchor,
        constraint_names=names,
        k=k,
        perspective_weight=1.0 / gamma,
        return_reward=return_reward,
        metadata=metadata,
    )
    instance.validate()
    return instance


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        description=(
            "Generate a portable large factor-form Markowitz instance"
        )
    )
    argument_parser.add_argument("--output", type=Path, required=True)
    argument_parser.add_argument("--dimension", type=int, default=100000)
    argument_parser.add_argument("--rank", type=int, default=50)
    argument_parser.add_argument("--k", type=int, default=100)
    argument_parser.add_argument(
        "--gamma-scale",
        type=float,
        choices=(1.0, 100.0),
        default=100.0,
        help="sets gamma = gamma_scale / sqrt(d), as in the paper",
    )
    argument_parser.add_argument(
        "--regime",
        choices=("unconstrained", "constrained", "hybrid"),
        default="constrained",
    )
    argument_parser.add_argument("--seed", type=int, default=7)
    argument_parser.add_argument("--sectors", type=int, default=20)
    argument_parser.add_argument("--style-factors", type=int, default=4)
    argument_parser.add_argument(
        "--stress-constraints",
        type=int,
        default=5,
    )
    argument_parser.add_argument("--target-fraction", type=float, default=0.3)
    argument_parser.add_argument("--target-iterations", type=int, default=200)
    argument_parser.add_argument("--sector-band", type=float, default=0.025)
    argument_parser.add_argument("--style-band", type=float, default=0.10)
    argument_parser.add_argument("--stress-band", type=float, default=0.005)
    argument_parser.add_argument(
        "--annual-volatility",
        type=float,
        default=0.20,
    )
    argument_parser.add_argument(
        "--common-correlation",
        type=float,
        default=0.15,
    )
    argument_parser.add_argument(
        "--daily-returns-npy",
        type=Path,
        default=None,
        help=(
            "optional samples-by-assets daily-return matrix; activates "
            "the paper's historical SVD recipe"
        ),
    )
    argument_parser.add_argument(
        "--outlier-mode",
        choices=("cell", "row", "none"),
        default="cell",
    )
    argument_parser.add_argument("--overwrite", action="store_true")
    return argument_parser


def main(arguments: Optional[Sequence[str]] = None) -> None:
    args = parser().parse_args(arguments)
    daily_returns = (
        np.load(args.daily_returns_npy, mmap_mode="r")
        if args.daily_returns_npy is not None
        else None
    )
    dimension = (
        int(daily_returns.shape[1])
        if daily_returns is not None
        else args.dimension
    )
    instance = generate_instance(
        dimension=dimension,
        rank=args.rank,
        k=args.k,
        gamma_scale=args.gamma_scale,
        regime=args.regime,
        seed=args.seed,
        sectors=args.sectors,
        style_factors=args.style_factors,
        stress_constraints=args.stress_constraints,
        target_fraction=args.target_fraction,
        target_iterations=args.target_iterations,
        sector_band=args.sector_band,
        style_band=args.style_band,
        stress_band=args.stress_band,
        annual_volatility=args.annual_volatility,
        common_correlation=args.common_correlation,
        daily_returns=daily_returns,
        outlier_mode=args.outlier_mode,
    )
    bundle = save_instance_bundle(
        instance,
        args.output,
        overwrite=args.overwrite,
    )
    size_bytes = sum(
        file.stat().st_size for file in bundle.iterdir() if file.is_file()
    )
    report = {
        "bundle": str(bundle),
        "dimension": instance.dimension,
        "rank": instance.rank,
        "k": instance.k,
        "constraints": instance.rows,
        "constraint_nnz": int(instance.C.nnz),
        "size_gib": size_bytes / (1024.0**3),
        "gamma": instance.metadata["gamma"],
        "perspective_weight": instance.perspective_weight,
        "regime": args.regime,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
