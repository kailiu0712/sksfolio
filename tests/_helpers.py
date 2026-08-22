"""Small deterministic instances shared by correctness tests."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from sksfolio.relaxation import MarkowitzInstance


def small_instance(
    *,
    dimension: int = 24,
    perspective_weight: float = 20.0,
    return_reward: float = 0.4,
) -> MarkowitzInstance:
    rng = np.random.default_rng(413)
    factors = min(5, dimension - 1)
    factor_loadings = rng.normal(
        scale=0.03,
        size=(dimension, factors),
    )
    expected_returns = rng.uniform(0.002, 0.015, size=dimension)
    constraint_matrix = sparse.csr_matrix(
        np.vstack(
            [
                np.ones(dimension),
                np.r_[
                    np.ones(dimension // 2),
                    np.zeros(dimension - dimension // 2),
                ],
            ]
        )
    )
    anchor = np.full(dimension, 1.0 / dimension)
    instance = MarkowitzInstance(
        factor_loadings=factor_loadings,
        expected_returns=expected_returns,
        constraint_matrix=constraint_matrix,
        lower_bounds=np.array([1.0, 0.25]),
        upper_bounds=np.array([1.0, 0.75]),
        feasible_anchor=anchor,
        constraint_names=["budget", "sector"],
        k=min(6, dimension),
        perspective_weight=perspective_weight,
        return_reward=return_reward,
        metadata={"kind": "unit-test"},
    )
    instance.validate()
    return instance
