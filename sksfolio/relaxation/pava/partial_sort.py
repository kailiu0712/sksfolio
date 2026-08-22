"""Partial-top-k finite-breakpoint PAVA proximal primitive."""

from __future__ import annotations

import importlib
import os
from typing import Any

import numpy as np

from ._common import singleton_subtraction, validate


try:
    _native_pava = importlib.import_module("sksfolio._native_pava")
except (ImportError, OSError):
    _native_pava = None

_NATIVE_DISABLED = os.environ.get(
    "SKSFOLIO_DISABLE_NATIVE_PAVA",
    "",
).strip().lower() in {"1", "true", "yes", "on"}


def _mixed_pool_boundary(
    top: np.ndarray,
    tail: np.ndarray,
    gamma: float,
) -> float:
    """Locate the unique mixed pool by deterministic breakpoint pruning."""
    top_breakpoints = singleton_subtraction(top, gamma)
    smallest_top = float(np.min(top_breakpoints))
    if tail.size == 0:
        return smallest_top
    largest_tail = float(np.max(tail))
    if largest_tail <= smallest_top:
        return smallest_top

    switch = gamma
    top_at_switch = np.maximum(
        0.0,
        (gamma + 1.0) * switch - gamma * top,
    )
    active_tail = tail[tail > switch]
    value_at_switch = float(
        np.sum(top_at_switch)
        + gamma * np.sum(switch - active_tail)
    )

    certified_alpha = 0.0
    certified_beta = 0.0
    if value_at_switch > 0.0:
        top_candidate = top_breakpoints < switch
        tail_certified = tail >= switch
        certified_alpha = gamma * float(
            np.count_nonzero(tail_certified)
        )
        certified_beta = gamma * float(np.sum(tail[tail_certified]))

        breakpoints = np.concatenate(
            (top_breakpoints[top_candidate], tail[~tail_certified])
        )
        kinds_top = np.concatenate(
            (
                np.ones(np.count_nonzero(top_candidate), dtype=bool),
                np.zeros(np.count_nonzero(~tail_certified), dtype=bool),
            )
        )
        alphas = np.concatenate(
            (
                np.full(
                    np.count_nonzero(top_candidate),
                    gamma + 1.0,
                ),
                np.full(
                    np.count_nonzero(~tail_certified),
                    gamma,
                ),
            )
        )
        betas = np.concatenate(
            (
                gamma * top[top_candidate],
                gamma * tail[~tail_certified],
            )
        )
    elif value_at_switch < 0.0:
        top_certified = top_breakpoints <= switch
        tail_candidate = tail > switch
        certified_alpha = float(np.count_nonzero(top_certified))
        certified_beta = float(
            np.sum(top[top_certified] - 1.0)
        )

        breakpoints = np.concatenate(
            (top_breakpoints[~top_certified], tail[tail_candidate])
        )
        kinds_top = np.concatenate(
            (
                np.ones(np.count_nonzero(~top_certified), dtype=bool),
                np.zeros(np.count_nonzero(tail_candidate), dtype=bool),
            )
        )
        alphas = np.ones(breakpoints.size)
        betas = np.concatenate(
            (
                top[~top_certified] - 1.0,
                tail[tail_candidate],
            )
        )
    else:
        return switch

    while breakpoints.size:
        middle = breakpoints.size // 2
        pivot = float(np.partition(breakpoints, middle)[middle])
        lower = breakpoints < pivot
        equal = breakpoints == pivot
        upper = breakpoints > pivot

        active_top = lower & kinds_top
        active_tail = upper & ~kinds_top
        alpha = (
            certified_alpha
            + float(np.sum(alphas[active_top]))
            + float(np.sum(alphas[active_tail]))
        )
        beta = (
            certified_beta
            + float(np.sum(betas[active_top]))
            + float(np.sum(betas[active_tail]))
        )
        value = alpha * pivot - beta
        scale = 1.0 + abs(alpha * pivot) + abs(beta)
        if abs(value) <= 8.0 * np.finfo(float).eps * scale:
            return pivot

        if value < 0.0:
            newly_certified = (lower | equal) & kinds_top
            certified_alpha += float(
                np.sum(alphas[newly_certified])
            )
            certified_beta += float(np.sum(betas[newly_certified]))
            keep = upper
        else:
            newly_certified = (upper | equal) & ~kinds_top
            certified_alpha += float(
                np.sum(alphas[newly_certified])
            )
            certified_beta += float(np.sum(betas[newly_certified]))
            keep = lower

        breakpoints = breakpoints[keep]
        kinds_top = kinds_top[keep]
        alphas = alphas[keep]
        betas = betas[keep]

    if certified_alpha == 0.0:
        return smallest_top
    return float(certified_beta / certified_alpha)


def prox_python(
    argument: Any,
    gamma: float,
    k: int,
) -> np.ndarray:
    """Compute the proximal point after selecting only the largest ``k``."""
    values = validate(argument, gamma, k)
    positive = np.maximum(values, 0.0)
    dimension = positive.size
    if not np.any(positive):
        return np.zeros_like(positive)
    if k == dimension:
        return np.minimum(positive / (1.0 + gamma), 1.0)

    top_indices = np.argpartition(
        positive,
        dimension - k,
    )[dimension - k :]
    top_mask = np.zeros(dimension, dtype=bool)
    top_mask[top_indices] = True
    top = positive[top_indices]
    tail = positive[~top_mask]
    boundary = _mixed_pool_boundary(top, tail, gamma)

    result = np.maximum(positive - boundary, 0.0)
    result[top_indices] = top - np.maximum(
        singleton_subtraction(top, gamma),
        boundary,
    )
    np.maximum(result, 0.0, out=result)
    np.minimum(result, 1.0, out=result)
    return result


def native_available() -> bool:
    """Return whether the compiled partial-selection kernel is importable."""
    return _native_pava is not None


def native_enabled() -> bool:
    """Return whether calls use the compiled kernel in this process."""
    return native_available() and not _NATIVE_DISABLED


def prox_native(
    argument: Any,
    gamma: float,
    k: int,
) -> np.ndarray:
    """Compute the proximal point with the compiled C implementation."""
    if _native_pava is None:
        raise RuntimeError(
            "the native PAVA extension is unavailable; reinstall "
            "sksfolio from source or use prox_python"
        )
    values = np.ascontiguousarray(validate(argument, gamma, k))
    result = np.empty_like(values)
    _native_pava.partial_sort_into(
        values,
        float(gamma),
        int(k),
        result,
    )
    return result


def prox(
    argument: Any,
    gamma: float,
    k: int,
) -> np.ndarray:
    """Use native partial selection when available, otherwise Python."""
    if native_enabled():
        return prox_native(argument, gamma, k)
    return prox_python(argument, gamma, k)


__all__ = [
    "native_available",
    "native_enabled",
    "prox",
    "prox_native",
    "prox_python",
]
