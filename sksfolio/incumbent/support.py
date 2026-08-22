"""Fast support primitives for sparse incumbent heuristics.

The binary perspective proximal map is separable except for the cardinality
budget.  Its exact solution is therefore a shrink-and-clip followed by a
partial top-k selection.  General portfolio rows are deliberately absent
from this module: once those rows are included the binary proximal problem is
itself a cardinality-constrained quadratic program.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

import numpy as np


def _as_indices(
    values: Optional[Iterable[int]],
    dimension: int,
    name: str,
) -> np.ndarray:
    if values is None:
        return np.empty(0, dtype=np.int64)
    array = np.asarray(tuple(values), dtype=np.int64).reshape(-1)
    if array.size == 0:
        return array
    if np.any(array < 0) or np.any(array >= dimension):
        raise ValueError(f"{name} contains an out-of-range index")
    return np.unique(array)


def validate_branch_indices(
    dimension: int,
    k: int,
    required: Optional[Iterable[int]] = None,
    forbidden: Optional[Iterable[int]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize fixed-in and fixed-out indices for later BnB use."""
    required_array = _as_indices(required, dimension, "required_assets")
    forbidden_array = _as_indices(
        forbidden,
        dimension,
        "forbidden_assets",
    )
    if required_array.size > int(k):
        raise ValueError("more than k assets are required")
    if np.intersect1d(required_array, forbidden_array).size:
        raise ValueError("an asset cannot be both required and forbidden")
    if dimension - forbidden_array.size < required_array.size:
        raise ValueError("branch restrictions leave no admissible support")
    return required_array, forbidden_array


def top_k_indices(
    scores: Any,
    k: int,
    *,
    required: Optional[Sequence[int]] = None,
    forbidden: Optional[Sequence[int]] = None,
    positive_only: bool = False,
) -> np.ndarray:
    """Return a deterministic partial top-k support.

    Required entries are inserted first.  Ties among the partially selected
    entries are resolved by score and then by the original index.
    """
    values = np.asarray(scores, dtype=float).reshape(-1)
    dimension = values.size
    if not 0 <= int(k) <= dimension:
        raise ValueError("k must lie in {0, ..., dimension}")
    required_array, forbidden_array = validate_branch_indices(
        dimension,
        int(k),
        required,
        forbidden,
    )
    allowed = np.ones(dimension, dtype=bool)
    allowed[forbidden_array] = False
    allowed[required_array] = False
    candidates = np.flatnonzero(allowed & np.isfinite(values))
    if positive_only:
        candidates = candidates[values[candidates] > 0.0]
    slots = min(int(k) - required_array.size, candidates.size)
    if slots <= 0:
        return np.sort(required_array)
    if slots < candidates.size:
        candidate_values = values[candidates]
        threshold = float(
            np.partition(
                candidate_values,
                candidates.size - slots,
            )[candidates.size - slots]
        )
        strict = candidates[candidate_values > threshold]
        tied = np.sort(candidates[candidate_values == threshold])
        chosen = np.concatenate((strict, tied[: slots - strict.size]))
    else:
        chosen = candidates
    order = np.lexsort((chosen, -values[chosen]))
    support = np.concatenate((required_array, chosen[order]))
    return np.sort(np.unique(support)).astype(np.int64, copy=False)


def perspective_activations(
    x: Any,
    k: int,
    tolerance: float = 1e-14,
) -> np.ndarray:
    """Recover a canonical optimal selector of the perspective envelope.

    For a dense nonnegative point, the returned vector satisfies
    ``z_i = min(1, x_i / tau)`` and ``sum(z) = k``.  Locating ``tau`` needs
    only the largest k positive entries rather than a full sort.
    """
    vector = np.asarray(x, dtype=float).reshape(-1)
    dimension = vector.size
    if not 1 <= int(k) <= dimension:
        raise ValueError("k must lie in {1, ..., dimension}")
    if np.any(~np.isfinite(vector)) or np.min(vector) < -tolerance:
        raise ValueError("x must be finite and nonnegative")
    values = np.maximum(vector, 0.0)
    positive_indices = np.flatnonzero(values > tolerance)
    result = np.zeros(dimension, dtype=float)
    if positive_indices.size == 0:
        return result
    if positive_indices.size <= int(k):
        result[positive_indices] = 1.0
        return result

    positive = values[positive_indices]
    split = positive.size - int(k)
    top = np.sort(np.partition(positive, split)[split:])[::-1]
    tail_sum = float(np.sum(positive))
    capped = 0
    while (
        capped < int(k)
        and (int(k) - capped) * float(top[capped])
        >= tail_sum - tolerance
    ):
        tail_sum -= float(top[capped])
        capped += 1
    if capped >= int(k) or tail_sum <= tolerance:
        result[top_k_indices(values, int(k))] = 1.0
        return result
    tau = tail_sum / float(int(k) - capped)
    result = np.minimum(1.0, values / max(tau, tolerance))
    # Remove a tiny accumulation error without changing the ordering.
    total = float(np.sum(result))
    if abs(total - float(k)) > 50.0 * np.finfo(float).eps * dimension:
        result = rescale_marginals(result, int(k))
    return result


def binary_perspective_prox(
    v: Any,
    step: float,
    perspective_weight: float,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact binary-perspective prox without coupling constraints.

    Returns the proximal point, its selected support, and the per-coordinate
    improvement over setting that coordinate to zero.
    """
    vector = np.asarray(v, dtype=float).reshape(-1)
    if np.any(~np.isfinite(vector)):
        raise ValueError("v must be finite")
    if step < 0.0 or perspective_weight < 0.0:
        raise ValueError("step and perspective_weight must be nonnegative")
    if not 0 <= int(k) <= vector.size:
        raise ValueError("k must lie in {0, ..., dimension}")
    scale = 1.0 + float(step) * float(perspective_weight)
    candidate = np.clip(vector / scale, 0.0, 1.0)
    gains = (
        0.5 * vector * vector
        - 0.5 * (candidate - vector) ** 2
        - 0.5
        * float(step)
        * float(perspective_weight)
        * candidate
        * candidate
    )
    gains = np.maximum(gains, 0.0)
    support = top_k_indices(gains, int(k), positive_only=True)
    result = np.zeros_like(vector)
    result[support] = candidate[support]
    return result, support, gains


def _project_simplex(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        raise ValueError("cannot project an empty vector onto the simplex")
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    indices = np.arange(1, values.size + 1, dtype=float)
    active = ordered - cumulative / indices > 0.0
    rho = int(np.flatnonzero(active)[-1])
    threshold = cumulative[rho] / float(rho + 1)
    result = np.maximum(values - threshold, 0.0)
    result /= float(np.sum(result))
    return result


def sparse_simplex_prox(
    v: Any,
    step: float,
    perspective_weight: float,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact binary prox when the only coupling row is ``sum(x) = 1``."""
    vector = np.asarray(v, dtype=float).reshape(-1)
    if np.any(~np.isfinite(vector)):
        raise ValueError("v must be finite")
    if step < 0.0 or perspective_weight < 0.0:
        raise ValueError("step and perspective_weight must be nonnegative")
    if not 1 <= int(k) <= vector.size:
        raise ValueError("k must lie in {1, ..., dimension}")
    support = top_k_indices(vector, int(k))
    scale = 1.0 + float(step) * float(perspective_weight)
    result = np.zeros_like(vector)
    result[support] = _project_simplex(vector[support] / scale)
    active = support[result[support] > 0.0]
    return result, np.sort(active)


def rescale_marginals(weights: Any, cardinality: int) -> np.ndarray:
    """Scale nonnegative weights into [0, 1] with the requested sum."""
    values = np.maximum(np.asarray(weights, dtype=float).reshape(-1), 0.0)
    if np.any(~np.isfinite(values)):
        raise ValueError("weights must be finite")
    target = int(cardinality)
    if not 0 <= target <= values.size:
        raise ValueError("cardinality is out of range")
    if target == 0:
        return np.zeros_like(values)
    positive = values > 0.0
    if np.count_nonzero(positive) < target:
        values = values.copy()
        values[~positive] = 1.0
    if target == values.size:
        return np.ones_like(values)
    low = 0.0
    high = 1.0
    while float(np.sum(np.minimum(1.0, high * values))) < target:
        high *= 2.0
    for _ in range(60):
        middle = 0.5 * (low + high)
        total = float(np.sum(np.minimum(1.0, middle * values)))
        if total < target:
            low = middle
        else:
            high = middle
    result = np.minimum(1.0, high * values)
    # Place the final roundoff on one non-bound entry.
    error = float(target) - float(np.sum(result))
    free = np.flatnonzero((result > 1e-15) & (result < 1.0 - 1e-15))
    if free.size:
        index = int(free[0])
        result[index] = np.clip(result[index] + error, 0.0, 1.0)
    return result


def dependent_round_support(
    probabilities: Any,
    rng: np.random.Generator,
    cardinality: Optional[int] = None,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Pivotal rounding with a fixed cardinality and preserved marginals."""
    values = np.asarray(probabilities, dtype=float).reshape(-1).copy()
    if np.any(~np.isfinite(values)) or np.any(values < -tolerance):
        raise ValueError("probabilities must be finite and nonnegative")
    values = np.clip(values, 0.0, 1.0)
    target = (
        int(round(float(np.sum(values))))
        if cardinality is None
        else int(cardinality)
    )
    if not 0 <= target <= values.size:
        raise ValueError("cardinality is out of range")
    if abs(float(np.sum(values)) - target) > 1e-9:
        values = rescale_marginals(values, target)

    fractional = list(
        np.flatnonzero((values > tolerance) & (values < 1.0 - tolerance))
    )
    while len(fractional) >= 2:
        first = int(fractional.pop())
        second = int(fractional.pop())
        first_value = float(values[first])
        second_value = float(values[second])
        increase = min(1.0 - first_value, second_value)
        decrease = min(first_value, 1.0 - second_value)
        denominator = increase + decrease
        if denominator <= tolerance:
            values[first] = float(first_value >= 0.5)
            values[second] = float(second_value >= 0.5)
            continue
        if float(rng.random()) < decrease / denominator:
            values[first] = first_value + increase
            values[second] = second_value - increase
        else:
            values[first] = first_value - decrease
            values[second] = second_value + decrease
        for index in (first, second):
            if values[index] <= tolerance:
                values[index] = 0.0
            elif values[index] >= 1.0 - tolerance:
                values[index] = 1.0
            else:
                fractional.append(index)

    support = np.flatnonzero(values >= 1.0 - tolerance)
    if support.size != target:
        support = top_k_indices(values, target)
    return np.sort(support).astype(np.int64, copy=False)


__all__ = [
    "binary_perspective_prox",
    "dependent_round_support",
    "perspective_activations",
    "rescale_marginals",
    "sparse_simplex_prox",
    "top_k_indices",
    "validate_branch_indices",
]
