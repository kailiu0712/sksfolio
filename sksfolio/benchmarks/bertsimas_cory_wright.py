"""Benchmark profiles from Bertsimas and Cory-Wright.

The paper benchmarks an integer cardinality model.  ``sksfolio`` solves its
continuous perspective relaxation, so the profiles below reproduce the
dimensions and parameter grids, not the paper's integer runtimes.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterator, Tuple

from .instance_generator import generate_instance


PAPER_URL = "https://arxiv.org/pdf/1811.00138"
REFERENCE_CODE_URL = (
    "https://github.com/ryancorywright/SparsePortfolioSelection.jl"
)

K_VALUES = (10, 50, 100, 200)
GAMMA_SCALES = (1.0, 100.0)
REGIMES = ("unconstrained", "constrained")
CONSTRAINT_PROFILES = {
    "bcw": {
        "sectors": 0,
        "style_factors": 0,
        "stress_constraints": 0,
        "sector_band": 0.0,
        "style_band": 0.0,
        "stress_band": 0.0,
    },
    "standard": {
        "sectors": 20,
        "style_factors": 8,
        "stress_constraints": 15,
        "sector_band": 0.03,
        "style_band": 0.10,
        "stress_band": 0.005,
    },
    "many": {
        "sectors": 20,
        "style_factors": 40,
        "stress_constraints": 200,
        "sector_band": 0.03,
        "style_band": 0.10,
        "stress_band": 0.005,
    },
}

# Tables 3 and 4.
OR_LIBRARY_UNIVERSES: Dict[str, int] = {
    "port1": 31,
    "port2": 85,
    "port3": 89,
    "port4": 98,
    "port5": 225,
}
OR_LIBRARY_K_VALUES = (5, 10, 20)

# Tables 7--9.  The nominal dimensions label the market indices.  The faithful
# dimensions used by the authors' processed return panels are 499, 958, and
# 3162 assets respectively.
HISTORICAL_UNIVERSES: Dict[
    str,
    Tuple[int, int, Tuple[int, ...]],
] = {
    "sp500": (500, 499, (50, 100, 150, 200)),
    "russell1000": (1000, 958, (50, 100, 200, 300)),
    "wilshire5000": (3200, 3162, (100, 200, 500, 1000)),
}


@dataclass(frozen=True, order=True)
class BCWCase:
    """One paper-grid case.

    ``regime='unconstrained'`` means the paper's ``kappa = 1`` objective.
    ``regime='constrained'`` means ``kappa = 0`` plus its minimum-return
    constraint.
    """

    family: str
    universe: str
    dimension: int
    rank: int
    k: int
    gamma_scale: float
    regime: str

    @property
    def gamma(self) -> float:
        return self.gamma_scale / math.sqrt(self.dimension)

    @property
    def key(self) -> str:
        scale = int(self.gamma_scale)
        return (
            f"{self.family}-{self.universe}-n{self.dimension}"
            f"-r{self.rank}-k{self.k}-g{scale}-{self.regime}"
        )


def or_library_cases() -> Tuple[BCWCase, ...]:
    """Return the 30 parameter combinations in paper Tables 3 and 4."""
    cases = []
    for universe, dimension in OR_LIBRARY_UNIVERSES.items():
        for k in OR_LIBRARY_K_VALUES:
            for regime in REGIMES:
                cases.append(
                    BCWCase(
                        family="or_library",
                        universe=universe,
                        dimension=dimension,
                        rank=dimension,
                        k=k,
                        gamma_scale=100.0,
                        regime=regime,
                    )
                )
    return tuple(cases)


def historical_cases(
    *,
    processed_dimensions: bool = True,
) -> Tuple[BCWCase, ...]:
    """Return the 192 parameter combinations in paper Tables 7--9.

    The default uses the dimensions of the processed panels used by the
    authors.  Set ``processed_dimensions=False`` only to request the nominal
    market-index labels 500, 1000, and 3200 for synthetic scaling studies.
    """
    cases = []
    for universe, (
        nominal_dimension,
        processed_dimension,
        ranks,
    ) in HISTORICAL_UNIVERSES.items():
        dimension = (
            processed_dimension
            if processed_dimensions
            else nominal_dimension
        )
        for rank in ranks:
            for k in K_VALUES:
                for gamma_scale in GAMMA_SCALES:
                    for regime in REGIMES:
                        cases.append(
                            BCWCase(
                                family="historical",
                                universe=universe,
                                dimension=dimension,
                                rank=rank,
                                k=k,
                                gamma_scale=gamma_scale,
                                regime=regime,
                            )
                        )
    return tuple(cases)


def iter_paper_cases() -> Iterator[BCWCase]:
    """Iterate over all 222 source profiles.

    The OR-Library profiles require the original covariance data.  Only the
    192 historical shapes can be adapted by ``generate_synthetic_case``.
    """
    yield from or_library_cases()
    yield from historical_cases()


def generate_synthetic_case(
    case: BCWCase,
    *,
    seed: int = 7,
    target_iterations: int = 200,
    constraint_profile: str = "bcw",
    sectors: int | None = None,
    style_factors: int | None = None,
    stress_constraints: int | None = None,
    sector_band: float | None = None,
    style_band: float | None = None,
    stress_band: float | None = None,
):
    """Generate a factor instance on a historical paper-grid shape.

    This is a deterministic synthetic adaptation.  It is not a replacement
    for the return data used by the paper and refuses OR-Library cases, whose
    full covariance matrices should be loaded from the original data.
    """
    if case.family != "historical":
        raise ValueError(
            "OR-Library cases require the original port1--port5 data"
        )
    if constraint_profile not in CONSTRAINT_PROFILES:
        choices = ", ".join(CONSTRAINT_PROFILES)
        raise ValueError(
            f"constraint_profile must be one of: {choices}"
        )
    profile = dict(CONSTRAINT_PROFILES[constraint_profile])
    overrides = {
        "sectors": sectors,
        "style_factors": style_factors,
        "stress_constraints": stress_constraints,
        "sector_band": sector_band,
        "style_band": style_band,
        "stress_band": stress_band,
    }
    profile.update(
        {
            name: value
            for name, value in overrides.items()
            if value is not None
        }
    )
    instance = generate_instance(
        dimension=case.dimension,
        rank=case.rank,
        k=case.k,
        gamma_scale=case.gamma_scale,
        regime=case.regime,
        seed=seed,
        sectors=int(profile["sectors"]),
        style_factors=int(profile["style_factors"]),
        stress_constraints=int(profile["stress_constraints"]),
        target_fraction=0.3,
        target_iterations=target_iterations,
        sector_band=float(profile["sector_band"]),
        style_band=float(profile["style_band"]),
        stress_band=float(profile["stress_band"]),
        annual_volatility=0.20,
        common_correlation=0.15,
    )
    instance.metadata["constraint_profile"] = constraint_profile
    return instance


__all__ = [
    "BCWCase",
    "CONSTRAINT_PROFILES",
    "GAMMA_SCALES",
    "HISTORICAL_UNIVERSES",
    "K_VALUES",
    "OR_LIBRARY_K_VALUES",
    "OR_LIBRARY_UNIVERSES",
    "PAPER_URL",
    "REFERENCE_CODE_URL",
    "REGIMES",
    "generate_synthetic_case",
    "historical_cases",
    "iter_paper_cases",
    "or_library_cases",
]
