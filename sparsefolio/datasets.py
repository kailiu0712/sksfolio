"""Reproducible factor-model instances with nested exposure constraints.

The factor-data random sequence reproduces the original synthetic rootgap
testcase.  The new exposure construction is inspired by FlashFolio's exposure
types, not a reproduction of its unpublished numerical exposure matrices.
"""
from __future__ import annotations

import numpy as np

from .core import Problem


def _base_data(n: int, seed: int):
    """Exactly follow make_factor_returns(n, 2*n, q=5, idio_share=.5)."""
    rng = np.random.default_rng(seed)
    q, periods = 5, 2 * n
    loading = rng.standard_normal((n, q)) / np.sqrt(q)
    factor_variance = rng.uniform(0.5, 1.5, q)
    systematic_variance = loading**2 @ factor_variance
    specific_variance = systematic_variance * rng.uniform(0.5, 1.5, n)
    factors = rng.standard_normal((periods, q)) * np.sqrt(factor_variance)
    errors = rng.standard_normal((periods, n)) * np.sqrt(specific_variance)
    mean = rng.normal(0.0, 0.05, n)
    returns = mean + factors @ loading.T + errors
    mu = returns.mean(axis=0)
    B = (loading * np.sqrt(factor_variance)).T
    return B, -mu, specific_variance + 0.001, loading


def make_problem(n=100, k=10, exposures=10, band="tight", seed=0):
    """Construct one root relaxation, without objective autoscaling.

    All cases include budget and median expected-return rows.
    Exposure rows 0--9 are balanced synthetic sectors and subsequent rows are
    synthetic, correlated style directions, standardized to unit cross-sectional
    standard deviation.  These later rows are a scaling stress test, not 190
    claimed economic factors.  Tight and wide halfwidths are respectively
    (.02, .03) and (.10, .20) for sector weights and standardized styles.
    Every band is centered at the same k-sparse feasible anchor.  Rows are
    generated sequentially, ensuring nested prefixes across exposure counts.
    """
    n, k, exposures, seed = int(n), int(k), int(exposures), int(seed)
    if n < 10 or not 1 <= k <= n or exposures < 0:
        raise ValueError("Require n>=10, 1<=k<=n, and exposures>=0")
    if band not in ("tight", "wide", "inactive"):
        raise ValueError("band must be tight, wide, or inactive")
    B, c, gamma, loading = _base_data(n, seed)
    mu, target = -c, float(np.quantile(-c, 0.5))

    # Selecting from the upper half preserves feasibility without dependence on
    # exposure count or width.  Equal weights on the top-k names are sufficient
    # for the sizes in this study.  The rare k~n case uses a high-return tilt.
    support = np.argsort(mu, kind="stable")[-k:]
    anchor = np.zeros(n)
    anchor[support] = 1.0 / k
    if mu @ anchor < target:
        best = int(np.argmax(mu))
        mixing = (target - mu @ anchor) / (mu[best] - mu @ anchor)
        anchor *= 1.0 - min(1.0, mixing + 1e-12)
        anchor[best] += min(1.0, mixing + 1e-12)

    # Independent streams avoid changing the objective or anchor when exposure
    # prefixes are extended.  Balanced sectors ensure all ten rows are present.
    sector_rng = np.random.default_rng(np.random.SeedSequence([seed, 7101]))
    sector_id = np.empty(n, dtype=int)
    sector_id[sector_rng.permutation(n)] = np.arange(n) % 10
    style_rng = np.random.default_rng(np.random.SeedSequence([seed, 7102]))
    standardized_loadings = loading - loading.mean(axis=0)
    standardized_loadings /= standardized_loadings.std(axis=0)

    sector_width, style_width = ((0.02, 0.03) if band == "tight" else (0.10, 0.20))
    rows = [np.ones(n), mu.copy()]
    lower, upper = [1.0, target], [1.0, np.inf]
    row_types = ["budget", "mean_return"]
    for j in range(exposures):
        if j < 10:
            row = (sector_id == j).astype(float)
            center = float(row @ anchor)
            lo, hi = max(0.0, center - sector_width), min(1.0, center + sector_width)
            row_type = "sector"
        else:
            mixing = style_rng.standard_normal(5)
            mixing /= np.linalg.norm(mixing)
            row = standardized_loadings @ mixing + 0.35 * style_rng.standard_normal(n)
            row -= row.mean()
            row /= row.std()
            center = float(row @ anchor)
            lo, hi = center - style_width, center + style_width
            row_type = "synthetic_style_stress"
        if band == "inactive":
            # Every simplex portfolio already satisfies these bounds. This
            # control adds rows without changing the feasible set or optimum.
            lo, hi = float(np.min(row))-0.01, float(np.max(row))+0.01
        rows.append(row)
        lower.append(lo)
        upper.append(hi)
        row_types.append(row_type)

    raw_K = np.ascontiguousarray(rows, dtype=float)
    raw_lo, raw_hi = np.array(lower), np.array(upper)
    row_scale = np.sqrt(n) / np.linalg.norm(raw_K, axis=1)
    K = np.ascontiguousarray(raw_K * row_scale[:, None])
    lo, hi = raw_lo * row_scale, raw_hi * row_scale
    metadata = dict(n=n, k=k, exposures=exposures, constraints=exposures + 2,
                    band=band, seed=seed, q=5, T=2*n, idio_share=0.5,
                    l2=0.001, kappa=1.0, lam=1.0, return_quantile=0.5,
                    return_target=target, lower_buyin=0.0, upper_buyin=1.0,
                    sector_halfwidth=sector_width, style_halfwidth=style_width,
                    row_types=row_types, sector_id=sector_id.tolist(),
                    anchor_support=support.tolist(), objective_autoscaling=False,
                    base_testcase="sksfolio/bench/suites/rootgap.yaml, factor diagonal, n=50,k=5,idio_share=.5,l2=.001,seed=0",
                    exposure_source="FlashFolio-style exposure classes; numerical rows and bounds are new synthetic experiments",
                    paper_url="https://arxiv.org/html/2604.22625v1")
    p = Problem(B=B, c=c, gamma=gamma, u=np.ones(n), k=k, K=K,
                lo=lo, hi=hi, raw_K=raw_K, raw_lo=raw_lo, raw_hi=raw_hi,
                row_scale=row_scale, anchor=anchor, metadata=metadata)
    if max(np.max(raw_lo - raw_K @ anchor), np.max(raw_K @ anchor - raw_hi)) > 1e-12:
        raise AssertionError("Constructed anchor is infeasible")
    return p
