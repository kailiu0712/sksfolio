"""Complete-sort pool-adjacent-violators proximal oracle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._common import validate


@dataclass
class _Pool:
    start: int
    stop: int
    selected: int
    total: float
    subtraction: float


def _pool(
    start: int,
    stop: int,
    selected: int,
    total: float,
    gamma: float,
) -> _Pool:
    size = stop - start
    denominator = gamma * size + selected
    if total <= denominator:
        subtraction = (gamma / denominator) * total
    else:
        subtraction = (total - selected) / size
    return _Pool(
        start=start,
        stop=stop,
        selected=selected,
        total=total,
        subtraction=float(subtraction),
    )


def prox(
    argument: Any,
    gamma: float,
    k: int,
) -> np.ndarray:
    """Compute the long-only proximal point using a complete stable sort."""
    values = validate(argument, gamma, k)
    positive = np.maximum(values, 0.0)
    if not np.any(positive):
        return np.zeros_like(positive)
    if k == positive.size:
        return np.minimum(positive / (1.0 + gamma), 1.0)

    order = np.argsort(-positive, kind="stable")
    ordered = positive[order]
    pools: list[_Pool] = []
    for position, magnitude in enumerate(ordered):
        pools.append(
            _pool(
                position,
                position + 1,
                int(position < k),
                float(magnitude),
                gamma,
            )
        )
        while (
            len(pools) >= 2
            and pools[-2].subtraction < pools[-1].subtraction
        ):
            right = pools.pop()
            left = pools.pop()
            pools.append(
                _pool(
                    left.start,
                    right.stop,
                    left.selected + right.selected,
                    left.total + right.total,
                    gamma,
                )
            )

    subtraction = np.empty_like(ordered)
    for pool in pools:
        subtraction[pool.start : pool.stop] = pool.subtraction
    ordered_result = ordered - subtraction
    result = np.empty_like(positive)
    result[order] = ordered_result
    np.maximum(result, 0.0, out=result)
    np.minimum(result, 1.0, out=result)
    return result


__all__ = ["prox"]
