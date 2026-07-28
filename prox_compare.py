#!/usr/bin/env python3
"""Compare two constrained-prox solvers with the same Gurobi model.

The function used in this experiment is

    G(x) = min_z 0.5 * sum_i x_i^2 / z_i
           s.t. 0 <= z <= 1, sum(z) <= k, |x_i| <= M z_i.

Both custom methods solve the nonnegative multiplier dual of

    min_x G(x) + ||x - v||^2 / (2 gamma)  s.t. A x <= b.

The base prox of G and its fixed-cell Jacobian-vector product are computed
by generalized PAVA.  Only the standard library is needed by the custom
solvers; gurobipy is needed for the reference solve.

The default benchmark has d = 2000, m = 50, and k = 400.  For example,
``--d 10000 --m 100 --k 2000 --row-nonzeros 64`` creates a larger case.
Use ``--benchmark-pava --pava-only`` to compare full sorting with the
corrected top-k-only implementation inspired by Haotian Wu's note.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

Vector = List[float]
Matrix = List[Vector]
_PIVOT_RNG = random.SystemRandom()


def dot(x: Sequence[float], y: Sequence[float]) -> float:
    return sum(xi * yi for xi, yi in zip(x, y))


def norm2(x: Sequence[float]) -> float:
    return math.sqrt(dot(x, x))


def norm_inf(x: Sequence[float]) -> float:
    return max((abs(xi) for xi in x), default=0.0)


def matvec(a: Matrix, x: Sequence[float]) -> Vector:
    return [dot(row, x) for row in a]


def tmatvec(a: Matrix, y: Sequence[float]) -> Vector:
    if not a:
        return []
    result = [0.0] * len(a[0])
    for yi, row in zip(y, a):
        for j, aij in enumerate(row):
            result[j] += aij * yi
    return result


def add_scaled(x: Sequence[float], alpha: float, y: Sequence[float]) -> Vector:
    return [xi + alpha * yi for xi, yi in zip(x, y)]


def positive_part(x: Sequence[float]) -> Vector:
    return [max(0.0, xi) for xi in x]


def max_linear_violation(a: Matrix, b: Sequence[float], x: Sequence[float]) -> float:
    return max((max(0.0, axi - bi) for axi, bi in zip(matvec(a, x), b)), default=0.0)


def spectral_norm_squared_upper_bound(a: Matrix) -> float:
    """Gershgorin upper bound for lambda_max(A A^T)."""
    gram = [[dot(row_i, row_j) for row_j in a] for row_i in a]
    return max((sum(abs(value) for value in row) for row in gram), default=0.0)


@dataclass
class Pool:
    start: int
    end: int
    size: int
    selected: int
    total: float
    subtraction: float
    kappa: float
    branch: str


@dataclass
class PavaState:
    u: Vector
    x: Vector
    y: Vector
    gamma: float
    k: int
    m_bound: float
    permutation: List[int]
    signs: Vector
    pools: List[Pool]
    moreau_value: float
    conjugate_value: float
    zero_jacobian_indices: Optional[List[int]] = None

    def jvp(self, h: Sequence[float]) -> Vector:
        """Apply one fixed-cell generalized Jacobian of prox_{gamma G}."""
        signed_sorted = [
            self.signs[j] * h[self.permutation[j]] for j in range(len(self.permutation))
        ]
        correction_sorted = [0.0] * len(self.permutation)
        for pool in self.pools:
            pool_sum = sum(signed_sorted[pool.start : pool.end])
            value = pool.kappa * pool_sum
            for j in range(pool.start, pool.end):
                correction_sorted[j] = self.signs[j] * value
        correction = [0.0] * len(self.permutation)
        for j, original_index in enumerate(self.permutation):
            correction[original_index] = correction_sorted[j]
        result = [hi - ci for hi, ci in zip(h, correction)]
        if self.zero_jacobian_indices is not None:
            for i in self.zero_jacobian_indices:
                result[i] = 0.0
        return result

    def signature(self) -> Tuple[Tuple[int, ...], Tuple[Tuple[int, int, str], ...]]:
        return (
            tuple(self.permutation),
            tuple((pool.start, pool.end, pool.branch) for pool in self.pools),
        )


def _make_pool(
    start: int,
    end: int,
    selected: int,
    total: float,
    gamma: float,
    m_bound: float,
) -> Pool:
    size = end - start
    denominator = gamma * size + selected
    if total <= m_bound * denominator:
        subtraction = gamma * total / denominator
        kappa = gamma / denominator
        branch = "quadratic"
    else:
        subtraction = (total - m_bound * selected) / size
        kappa = 1.0 / size
        branch = "linear"
    return Pool(start, end, size, selected, total, subtraction, kappa, branch)


def _assemble_pava_state(
    u: Sequence[float],
    gamma: float,
    k: int,
    m_bound: float,
    permutation: List[int],
    signs: Vector,
    pools: List[Pool],
    implicit_tail_start: Optional[int] = None,
) -> PavaState:
    subtraction_sorted = [0.0] * len(u)
    for pool in pools:
        for j in range(pool.start, pool.end):
            subtraction_sorted[j] = pool.subtraction
    zero_jacobian_indices: Optional[List[int]] = None
    if implicit_tail_start is not None:
        zero_jacobian_indices = []
        for j in range(implicit_tail_start, len(permutation)):
            original_index = permutation[j]
            subtraction_sorted[j] = abs(u[original_index])
            zero_jacobian_indices.append(original_index)

    x = [0.0] * len(u)
    y = [0.0] * len(u)
    y_sorted = [0.0] * len(u)
    for j, original_index in enumerate(permutation):
        signed_subtraction = signs[j] * subtraction_sorted[j]
        x[original_index] = u[original_index] - signed_subtraction
        y[original_index] = signed_subtraction / gamma
        y_sorted[j] = signs[j] * subtraction_sorted[j] / gamma

    def huber(value: float) -> float:
        magnitude = abs(value)
        if magnitude <= m_bound:
            return 0.5 * magnitude * magnitude
        return m_bound * magnitude - 0.5 * m_bound * m_bound

    conjugate_value = sum(huber(y_sorted[j]) for j in range(k))
    moreau_value = dot(y, u) - 0.5 * gamma * dot(y, y) - conjugate_value
    return PavaState(
        list(u),
        x,
        y,
        gamma,
        k,
        m_bound,
        permutation,
        signs,
        pools,
        moreau_value,
        conjugate_value,
        zero_jacobian_indices,
    )


def perspective_prox_state(
    u: Sequence[float],
    gamma: float,
    k: int,
    m_bound: float,
) -> PavaState:
    """PAVA oracle for prox_{gamma G}, its envelope, and a fixed-cell JVP."""
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")
    if not 1 <= k <= len(u):
        raise ValueError("k must be in {1, ..., d}")
    if m_bound <= 0.0:
        raise ValueError("M must be positive")

    permutation = sorted(range(len(u)), key=lambda i: (-abs(u[i]), i))
    magnitudes = [abs(u[i]) for i in permutation]
    signs = [1.0 if u[i] >= 0.0 else -1.0 for i in permutation]
    pools: List[Pool] = []

    for j, magnitude in enumerate(magnitudes):
        selected = 1 if j < k else 0
        pools.append(_make_pool(j, j + 1, selected, magnitude, gamma, m_bound))
        while len(pools) >= 2 and pools[-2].subtraction < pools[-1].subtraction:
            right = pools.pop()
            left = pools.pop()
            pools.append(
                _make_pool(
                    left.start,
                    right.end,
                    left.selected + right.selected,
                    left.total + right.total,
                    gamma,
                    m_bound,
                )
            )

    return _assemble_pava_state(u, gamma, k, m_bound, permutation, signs, pools)


def _top_scalar_subtraction(
    magnitude: float,
    gamma: float,
    m_bound: float,
) -> float:
    if magnitude <= m_bound * (gamma + 1.0):
        return gamma * magnitude / (gamma + 1.0)
    return magnitude - m_bound


def _topk_boundary_root(
    top_indices: Sequence[int],
    tail_indices: Sequence[int],
    magnitudes: Sequence[float],
    gamma: float,
    m_bound: float,
) -> float:
    """Find the only possible mixed PAVA pool by breakpoint pruning."""
    top_breakpoints = {
        i: _top_scalar_subtraction(magnitudes[i], gamma, m_bound) for i in top_indices
    }
    if not tail_indices:
        return min(top_breakpoints.values())
    smallest_top = min(top_breakpoints.values())
    largest_tail = max(magnitudes[i] for i in tail_indices)
    if largest_tail <= smallest_top:
        return smallest_top

    huber_breakpoint = gamma * m_bound
    value_at_breakpoint = 0.0
    for i in top_indices:
        value_at_breakpoint += max(
            0.0,
            (gamma + 1.0) * huber_breakpoint - gamma * magnitudes[i],
        )
    for i in tail_indices:
        if magnitudes[i] > huber_breakpoint:
            value_at_breakpoint += gamma * (huber_breakpoint - magnitudes[i])

    # Candidate tuple: (kind, breakpoint, alpha, beta).
    # A top item is active above its breakpoint; a tail item is active below.
    candidates: List[Tuple[str, float, float, float]] = []
    certified_alpha = 0.0
    certified_beta = 0.0
    if value_at_breakpoint > 0.0:
        for i in top_indices:
            breakpoint = top_breakpoints[i]
            if breakpoint < huber_breakpoint:
                candidates.append(
                    (
                        "top",
                        breakpoint,
                        gamma + 1.0,
                        gamma * magnitudes[i],
                    )
                )
        for i in tail_indices:
            magnitude = magnitudes[i]
            if magnitude >= huber_breakpoint:
                certified_alpha += gamma
                certified_beta += gamma * magnitude
            else:
                candidates.append(("tail", magnitude, gamma, gamma * magnitude))
    elif value_at_breakpoint < 0.0:
        for i in top_indices:
            breakpoint = top_breakpoints[i]
            beta = magnitudes[i] - m_bound
            if breakpoint <= huber_breakpoint:
                certified_alpha += 1.0
                certified_beta += beta
            else:
                candidates.append(("top", breakpoint, 1.0, beta))
        for i in tail_indices:
            magnitude = magnitudes[i]
            if magnitude > huber_breakpoint:
                candidates.append(("tail", magnitude, 1.0, magnitude))
    else:
        return huber_breakpoint

    while candidates:
        pivot = candidates[_PIVOT_RNG.randrange(len(candidates))][1]
        low: List[Tuple[str, float, float, float]] = []
        equal: List[Tuple[str, float, float, float]] = []
        high: List[Tuple[str, float, float, float]] = []
        pivot_alpha = certified_alpha
        pivot_beta = certified_beta

        for item in candidates:
            kind, breakpoint, alpha, beta = item
            if breakpoint < pivot:
                low.append(item)
                if kind == "top":
                    pivot_alpha += alpha
                    pivot_beta += beta
            elif breakpoint > pivot:
                high.append(item)
                if kind == "tail":
                    pivot_alpha += alpha
                    pivot_beta += beta
            else:
                equal.append(item)

        value = pivot_alpha * pivot - pivot_beta
        scale = 1.0 + abs(pivot_alpha * pivot) + abs(pivot_beta)
        if abs(value) <= 1e-15 * scale:
            return pivot
        if value < 0.0:
            for kind, _, alpha, beta in low + equal:
                if kind == "top":
                    certified_alpha += alpha
                    certified_beta += beta
            candidates = high
        else:
            for kind, _, alpha, beta in high + equal:
                if kind == "tail":
                    certified_alpha += alpha
                    certified_beta += beta
            candidates = low

    if certified_alpha == 0.0:
        return smallest_top
    return certified_beta / certified_alpha


def perspective_prox_state_topk(
    u: Sequence[float],
    gamma: float,
    k: int,
    m_bound: float,
) -> PavaState:
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")
    if not 1 <= k <= len(u):
        raise ValueError("k must be in {1, ..., d}")
    if m_bound <= 0.0:
        raise ValueError("M must be positive")

    magnitudes = [abs(value) for value in u]
    top_indices = heapq.nlargest(
        k,
        range(len(u)),
        key=lambda i: (magnitudes[i], -i),
    )
    top_set = set(top_indices)
    tail_indices = [i for i in range(len(u)) if i not in top_set]
    boundary_value = _topk_boundary_root(
        top_indices,
        tail_indices,
        magnitudes,
        gamma,
        m_bound,
    )
    mixed_top = [
        i
        for i in top_indices
        if _top_scalar_subtraction(magnitudes[i], gamma, m_bound) < boundary_value
    ]
    active_tail = [i for i in tail_indices if magnitudes[i] > boundary_value]
    mixed_top_set = set(mixed_top)
    active_tail_set = set(active_tail)
    singleton_top = [i for i in top_indices if i not in mixed_top_set]
    inactive_tail = [i for i in tail_indices if i not in active_tail_set]
    permutation = singleton_top + mixed_top + active_tail + inactive_tail
    signs = [1.0 if u[i] >= 0.0 else -1.0 for i in permutation]
    pools: List[Pool] = []

    position = 0
    for i in singleton_top:
        pools.append(
            _make_pool(
                position,
                position + 1,
                1,
                magnitudes[i],
                gamma,
                m_bound,
            )
        )
        position += 1

    if mixed_top or active_tail:
        mixed_indices = mixed_top + active_tail
        boundary = _make_pool(
            position,
            position + len(mixed_indices),
            len(mixed_top),
            sum(magnitudes[i] for i in mixed_indices),
            gamma,
            m_bound,
        )
        pools.append(boundary)
        position += len(mixed_indices)

    return _assemble_pava_state(
        u,
        gamma,
        k,
        m_bound,
        permutation,
        signs,
        pools,
        implicit_tail_start=position,
    )


def perspective_prox_state_dispatch(
    u: Sequence[float],
    gamma: float,
    k: int,
    m_bound: float,
    method: str = "auto",
) -> PavaState:
    if method == "auto":
        method = "topk" if 8 * k < len(u) else "full"
    if method == "topk":
        return perspective_prox_state_topk(u, gamma, k, m_bound)
    if method == "full":
        return perspective_prox_state(u, gamma, k, m_bound)
    raise ValueError("PAVA method must be one of: auto, full, topk")


def perspective_value(
    x: Sequence[float],
    k: int,
    m_bound: float,
) -> Tuple[float, Vector]:
    """Evaluate G(x) and return one minimizing z."""
    radii = [abs(xi) for xi in x]
    lower_sum = sum(radius / m_bound for radius in radii)
    if max(radii, default=0.0) > m_bound + 1e-10 or lower_sum > k + 1e-10:
        return math.inf, [math.nan] * len(x)

    positive = [i for i, radius in enumerate(radii) if radius > 0.0]
    z = [0.0] * len(x)
    if len(positive) <= k:
        for i in positive:
            z[i] = 1.0
    else:
        lower = 1.0 / m_bound
        upper = max(lower, 1.0)

        def z_sum(scale: float) -> float:
            return sum(min(1.0, scale * radius) for radius in radii)

        while z_sum(upper) < k:
            upper *= 2.0
        for _ in range(100):
            middle = 0.5 * (lower + upper)
            if z_sum(middle) < k:
                lower = middle
            else:
                upper = middle
        scale = upper
        for i in positive:
            z[i] = min(1.0, max(radii[i] / m_bound, scale * radii[i]))

    value = 0.0
    for radius, zi in zip(radii, z):
        if radius > 0.0:
            value += 0.5 * radius * radius / zi
    return value, z


def primal_objective(
    x: Sequence[float],
    v: Sequence[float],
    gamma: float,
    k: int,
    m_bound: float,
) -> float:
    g_value, _ = perspective_value(x, k, m_bound)
    return (
        g_value
        + 0.5
        * dot([xi - vi for xi, vi in zip(x, v)], [xi - vi for xi, vi in zip(x, v)])
        / gamma
    )


@dataclass
class DualEvaluation:
    multiplier: Vector
    state: PavaState
    x: Vector
    gradient: Vector
    value: float


def evaluate_dual(
    multiplier: Sequence[float],
    a: Matrix,
    b: Sequence[float],
    v: Sequence[float],
    gamma: float,
    k: int,
    m_bound: float,
    av_minus_b: Optional[Sequence[float]] = None,
    pava_method: str = "auto",
) -> DualEvaluation:
    at_multiplier = tmatvec(a, multiplier)
    shifted = [vi - gamma * ati for vi, ati in zip(v, at_multiplier)]
    state = perspective_prox_state_dispatch(
        shifted, gamma, k, m_bound, method=pava_method
    )
    ax = matvec(a, state.x)
    gradient = [axi - bi for axi, bi in zip(ax, b)]
    if av_minus_b is None:
        av_minus_b = [avi - bi for avi, bi in zip(matvec(a, v), b)]
    value = (
        state.moreau_value
        + dot(multiplier, av_minus_b)
        - 0.5 * gamma * dot(at_multiplier, at_multiplier)
    )
    return DualEvaluation(list(multiplier), state, state.x, gradient, value)


@dataclass
class Certificate:
    x_feasible: Vector
    theta: float
    primal_value: float
    dual_value: float
    gap: float
    violation: float


def strict_zero_anchor_certificate(
    evaluation: DualEvaluation,
    a: Matrix,
    b: Sequence[float],
    v: Sequence[float],
    gamma: float,
    k: int,
    m_bound: float,
) -> Certificate:
    if any(bi <= 0.0 for bi in b):
        raise ValueError("The zero-anchor certificate requires b > 0 componentwise")
    theta = 0.0
    for residual, slack in zip(evaluation.gradient, b):
        violation = max(0.0, residual)
        theta = max(theta, violation / (violation + slack))
    x_feasible = [(1.0 - theta) * xi for xi in evaluation.x]
    primal_value = primal_objective(x_feasible, v, gamma, k, m_bound)
    raw_gap = primal_value - evaluation.value
    numerical_scale = max(1.0, abs(primal_value), abs(evaluation.value))
    if raw_gap < -1e-10 * numerical_scale:
        raise ArithmeticError("The computed primal-dual gap is materially negative")
    gap = max(0.0, raw_gap)
    return Certificate(
        x_feasible,
        theta,
        primal_value,
        evaluation.value,
        gap,
        max_linear_violation(a, b, x_feasible),
    )


@dataclass
class TracePoint:
    iteration: int
    pava_calls: int
    seconds: float
    objective: float
    dual_value: float
    certificate_gap: float
    violation: float
    kkt_residual: float


@dataclass
class SolverResult:
    method: str
    x: Vector
    multiplier: Optional[Vector]
    objective: float
    certificate_gap: Optional[float]
    kkt_residual: Optional[float]
    violation: float
    iterations: int
    pava_calls: int
    seconds: float
    status: str
    trace: Optional[List[TracePoint]] = None


class ProgressBar:
    def __init__(self, label: str, total: int, enabled: bool = True) -> None:
        self.label = label
        self.total = max(1, total)
        self.enabled = enabled
        self.width = 28
        self.last_draw = 0.0

    def update(
        self,
        current: int,
        *,
        gap: Optional[float] = None,
        residual: Optional[float] = None,
        pava_calls: Optional[int] = None,
        force: bool = False,
    ) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        if not force and current < self.total and now - self.last_draw < 0.08:
            return
        self.last_draw = now
        ratio = min(1.0, max(0.0, current / self.total))
        filled = min(self.width, int(round(self.width * ratio)))
        bar = "#" * filled + "." * (self.width - filled)
        details = []
        if gap is not None and math.isfinite(gap):
            details.append(f"gap={gap:.2e}")
        if residual is not None and math.isfinite(residual):
            details.append(f"res={residual:.2e}")
        if pava_calls is not None:
            details.append(f"pava={pava_calls}")
        detail_text = ""
        if details:
            detail_text = "  " + "  ".join(details)
        print(
            f"\r{self.label:<18} [{bar}] {current:>5}/{self.total:<5}{detail_text}",
            end="",
            file=sys.stderr,
            flush=True,
        )

    def finish(
        self,
        current: int,
        *,
        gap: Optional[float] = None,
        residual: Optional[float] = None,
        pava_calls: Optional[int] = None,
    ) -> None:
        self.update(
            current,
            gap=gap,
            residual=residual,
            pava_calls=pava_calls,
            force=True,
        )
        if self.enabled:
            print(file=sys.stderr, flush=True)


def restarted_dual_fista(
    a: Matrix,
    b: Sequence[float],
    v: Sequence[float],
    gamma: float,
    k: int,
    m_bound: float,
    tolerance: float = 1e-10,
    restart_length: int = 25,
    max_iterations: int = 20000,
    lipschitz: Optional[float] = None,
    pava_method: str = "auto",
    show_progress: bool = True,
) -> SolverResult:
    if lipschitz is None:
        lipschitz = gamma * spectral_norm_squared_upper_bound(a)
    if lipschitz <= 0.0:
        state = perspective_prox_state_dispatch(
            v, gamma, k, m_bound, method=pava_method
        )
        objective = primal_objective(state.x, v, gamma, k, m_bound)
        return SolverResult(
            "Restarted FISTA",
            state.x,
            [],
            objective,
            0.0,
            0.0,
            0.0,
            0,
            1,
            0.0,
            "converged",
            [
                TracePoint(
                    iteration=0,
                    pava_calls=1,
                    seconds=0.0,
                    objective=objective,
                    dual_value=objective,
                    certificate_gap=0.0,
                    violation=0.0,
                    kkt_residual=0.0,
                )
            ],
        )

    start = time.perf_counter()
    multiplier = [0.0] * len(b)
    extrapolated = multiplier[:]
    momentum = 1.0
    pava_calls = 0
    best: Optional[Tuple[Certificate, DualEvaluation]] = None
    av_minus_b = [avi - bi for avi, bi in zip(matvec(a, v), b)]
    trace: List[TracePoint] = []
    progress = ProgressBar("Restarted FISTA", max_iterations, enabled=show_progress)

    for iteration in range(1, max_iterations + 1):
        evaluation_y = evaluate_dual(
            extrapolated,
            a,
            b,
            v,
            gamma,
            k,
            m_bound,
            av_minus_b=av_minus_b,
            pava_method=pava_method,
        )
        pava_calls += 1
        multiplier_next = positive_part(
            [
                yi + gradient_i / lipschitz
                for yi, gradient_i in zip(extrapolated, evaluation_y.gradient)
            ]
        )
        momentum_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * momentum * momentum))
        extrapolated_next = [
            next_i + (momentum - 1.0) * (next_i - old_i) / momentum_next
            for next_i, old_i in zip(multiplier_next, multiplier)
        ]
        multiplier = multiplier_next
        extrapolated = extrapolated_next
        momentum = momentum_next

        if iteration % restart_length == 0 or iteration == max_iterations:
            extrapolated = multiplier[:]
            momentum = 1.0
            evaluation = evaluate_dual(
                multiplier,
                a,
                b,
                v,
                gamma,
                k,
                m_bound,
                av_minus_b=av_minus_b,
                pava_method=pava_method,
            )
            pava_calls += 1
            certificate = strict_zero_anchor_certificate(
                evaluation, a, b, v, gamma, k, m_bound
            )
            if best is None or certificate.gap < best[0].gap:
                best = (certificate, evaluation)
            residual = [
                li - max(0.0, li + gi / lipschitz)
                for li, gi in zip(multiplier, evaluation.gradient)
            ]
            residual_norm = norm_inf(residual)
            trace.append(
                TracePoint(
                    iteration=iteration,
                    pava_calls=pava_calls,
                    seconds=time.perf_counter() - start,
                    objective=certificate.primal_value,
                    dual_value=evaluation.value,
                    certificate_gap=certificate.gap,
                    violation=certificate.violation,
                    kkt_residual=residual_norm,
                )
            )
            progress.update(
                iteration,
                gap=certificate.gap,
                residual=residual_norm,
                pava_calls=pava_calls,
            )
            if certificate.gap <= tolerance:
                seconds = time.perf_counter() - start
                progress.finish(
                    iteration,
                    gap=certificate.gap,
                    residual=residual_norm,
                    pava_calls=pava_calls,
                )
                return SolverResult(
                    "Restarted FISTA",
                    certificate.x_feasible,
                    multiplier,
                    certificate.primal_value,
                    certificate.gap,
                    residual_norm,
                    certificate.violation,
                    iteration,
                    pava_calls,
                    seconds,
                    "converged",
                    trace,
                )

    assert best is not None
    certificate, evaluation = best
    residual = [
        li - max(0.0, li + gi / lipschitz)
        for li, gi in zip(evaluation.multiplier, evaluation.gradient)
    ]
    residual_norm = norm_inf(residual)
    progress.finish(
        max_iterations,
        gap=certificate.gap,
        residual=residual_norm,
        pava_calls=pava_calls,
    )
    return SolverResult(
        "Restarted FISTA",
        certificate.x_feasible,
        evaluation.multiplier,
        certificate.primal_value,
        certificate.gap,
        residual_norm,
        certificate.violation,
        max_iterations,
        pava_calls,
        time.perf_counter() - start,
        "iteration limit",
        trace,
    )


def solve_dense(matrix: Matrix, right_hand_side: Sequence[float]) -> Vector:
    """Gaussian elimination with partial pivoting."""
    n = len(right_hand_side)
    augmented = [row[:] + [rhs] for row, rhs in zip(matrix, right_hand_side)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-14:
            raise ArithmeticError("singular Newton matrix")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        for j in range(column, n + 1):
            augmented[column][j] /= pivot_value
        for row in range(column + 1, n):
            factor = augmented[row][column]
            if factor == 0.0:
                continue
            for j in range(column, n + 1):
                augmented[row][j] -= factor * augmented[column][j]
    solution = [0.0] * n
    for row in range(n - 1, -1, -1):
        solution[row] = augmented[row][n] - sum(
            augmented[row][j] * solution[j] for j in range(row + 1, n)
        )
    return solution


def globalized_semismooth_newton(
    a: Matrix,
    b: Sequence[float],
    v: Sequence[float],
    gamma: float,
    k: int,
    m_bound: float,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
    lipschitz: Optional[float] = None,
    pava_method: str = "auto",
    show_progress: bool = True,
) -> SolverResult:
    if lipschitz is None:
        lipschitz = gamma * spectral_norm_squared_upper_bound(a)
    c_parameter = 1.0 / lipschitz
    start = time.perf_counter()
    multiplier = [0.0] * len(b)
    pava_calls = 0
    best: Optional[Tuple[Certificate, DualEvaluation, float]] = None
    av_minus_b = [avi - bi for avi, bi in zip(matvec(a, v), b)]
    trace: List[TracePoint] = []
    progress = ProgressBar(
        "Semismooth Newton", max_iterations, enabled=show_progress
    )

    for iteration in range(max_iterations + 1):
        evaluation = evaluate_dual(
            multiplier,
            a,
            b,
            v,
            gamma,
            k,
            m_bound,
            av_minus_b=av_minus_b,
            pava_method=pava_method,
        )
        pava_calls += 1
        certificate = strict_zero_anchor_certificate(
            evaluation, a, b, v, gamma, k, m_bound
        )
        residual = [
            li - max(0.0, li + c_parameter * gi)
            for li, gi in zip(multiplier, evaluation.gradient)
        ]
        residual_norm = norm_inf(residual)
        trace.append(
            TracePoint(
                iteration=iteration,
                pava_calls=pava_calls,
                seconds=time.perf_counter() - start,
                objective=certificate.primal_value,
                dual_value=evaluation.value,
                certificate_gap=certificate.gap,
                violation=certificate.violation,
                kkt_residual=residual_norm,
            )
        )
        progress.update(
            iteration,
            gap=certificate.gap,
            residual=residual_norm,
            pava_calls=pava_calls,
        )
        if best is None or certificate.gap < best[0].gap:
            best = (certificate, evaluation, residual_norm)
        if certificate.gap <= tolerance:
            progress.finish(
                iteration,
                gap=certificate.gap,
                residual=residual_norm,
                pava_calls=pava_calls,
            )
            return SolverResult(
                "Semismooth Newton",
                certificate.x_feasible,
                multiplier,
                certificate.primal_value,
                certificate.gap,
                residual_norm,
                certificate.violation,
                iteration,
                pava_calls,
                time.perf_counter() - start,
                "converged",
                trace,
            )
        if iteration == max_iterations:
            break

        projected_argument = [
            li + c_parameter * gi for li, gi in zip(multiplier, evaluation.gradient)
        ]
        tie_tolerance = 1e-12 * (1.0 + norm_inf(projected_argument))
        projection_derivative = [
            1.0 if value > tie_tolerance else 0.0 for value in projected_argument
        ]

        hessian = [[0.0] * len(b) for _ in b]
        for column, row in enumerate(a):
            j_row = evaluation.state.jvp(row)
            for i, row_i in enumerate(a):
                hessian[i][column] = gamma * dot(row_i, j_row)

        newton_matrix = [[0.0] * len(b) for _ in b]
        for i in range(len(b)):
            for j in range(len(b)):
                identity_part = (
                    1.0 if i == j and projection_derivative[i] == 0.0 else 0.0
                )
                newton_matrix[i][j] = (
                    identity_part
                    + c_parameter * projection_derivative[i] * hessian[i][j]
                )

        direction: Optional[Vector] = None
        regularization = 0.0
        for _ in range(8):
            regularized = [row[:] for row in newton_matrix]
            for i in range(len(b)):
                regularized[i][i] += regularization
            try:
                direction = solve_dense(regularized, [-ri for ri in residual])
                break
            except ArithmeticError:
                regularization = (
                    1e-12 if regularization == 0.0 else 10.0 * regularization
                )

        projected_gradient_step = [
            max(0.0, li + gi / lipschitz) - li
            for li, gi in zip(multiplier, evaluation.gradient)
        ]
        if direction is None:
            search_direction = projected_gradient_step
        else:
            projected_newton_step = [
                max(0.0, li + di) - li for li, di in zip(multiplier, direction)
            ]
            newton_slope = dot(evaluation.gradient, projected_newton_step)
            reference_slope = lipschitz * dot(
                projected_gradient_step, projected_gradient_step
            )
            if newton_slope >= 1e-4 * reference_slope:
                search_direction = projected_newton_step
            else:
                search_direction = projected_gradient_step

        slope = dot(evaluation.gradient, search_direction)
        step_size = 1.0
        accepted = False
        for _ in range(40):
            candidate = add_scaled(multiplier, step_size, search_direction)
            candidate_evaluation = evaluate_dual(
                candidate,
                a,
                b,
                v,
                gamma,
                k,
                m_bound,
                av_minus_b=av_minus_b,
                pava_method=pava_method,
            )
            pava_calls += 1
            if (
                candidate_evaluation.value
                >= evaluation.value + 1e-4 * step_size * slope
            ):
                multiplier = candidate
                accepted = True
                break
            step_size *= 0.5
        if not accepted:
            multiplier = positive_part(
                [li + gi / lipschitz for li, gi in zip(multiplier, evaluation.gradient)]
            )

    assert best is not None
    certificate, evaluation, residual_norm = best
    progress.finish(
        max_iterations,
        gap=certificate.gap,
        residual=residual_norm,
        pava_calls=pava_calls,
    )
    return SolverResult(
        "Semismooth Newton",
        certificate.x_feasible,
        evaluation.multiplier,
        certificate.primal_value,
        certificate.gap,
        residual_norm,
        certificate.violation,
        max_iterations,
        pava_calls,
        time.perf_counter() - start,
        "iteration limit",
        trace,
    )


def solve_with_gurobi(
    a: Matrix,
    b: Sequence[float],
    v: Sequence[float],
    gamma: float,
    k: int,
    m_bound: float,
    output_flag: int = 0,
    barrier_tolerance: float = 1e-9,
) -> SolverResult:
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as error:
        raise RuntimeError(
            "gurobipy is unavailable; run with Gurobi's Python interpreter"
        ) from error

    start = time.perf_counter()
    model = gp.Model("constrained_perspective_prox")
    model.Params.OutputFlag = output_flag
    model.Params.NonConvex = 0
    model.Params.Threads = 1
    model.Params.FeasibilityTol = 1e-9
    model.Params.OptimalityTol = 1e-9
    model.Params.BarConvTol = 1e-12
    model.Params.BarQCPConvTol = barrier_tolerance
    model.Params.NumericFocus = 3

    d = len(v)
    x = model.addVars(d, lb=-GRB.INFINITY, name="x")
    z = model.addVars(d, lb=0.0, ub=1.0, name="z")
    t = model.addVars(d, lb=0.0, name="t")
    for i in range(d):
        model.addQConstr(x[i] * x[i] <= t[i] * z[i], name=f"perspective_{i}")
        model.addConstr(x[i] <= m_bound * z[i], name=f"upper_link_{i}")
        model.addConstr(x[i] >= -m_bound * z[i], name=f"lower_link_{i}")
    model.addConstr(gp.quicksum(z[i] for i in range(d)) <= k, name="budget")
    for row_number, (row, rhs) in enumerate(zip(a, b)):
        model.addConstr(
            gp.quicksum(
                coefficient * x[i]
                for i, coefficient in enumerate(row)
                if coefficient != 0.0
            )
            <= rhs,
            name=f"linear_{row_number}",
        )

    objective = 0.5 * gp.quicksum(t[i] for i in range(d))
    objective += (
        0.5 / gamma * gp.quicksum((x[i] - v[i]) * (x[i] - v[i]) for i in range(d))
    )
    model.setObjective(objective, GRB.MINIMIZE)
    model.optimize()
    acceptable_statuses = {GRB.OPTIMAL, GRB.SUBOPTIMAL}
    if model.Status not in acceptable_statuses or model.SolCount == 0:
        raise RuntimeError(f"Gurobi terminated with status {model.Status}")

    x_value = [x[i].X for i in range(d)]
    domain_scale = 1.0
    max_abs_x = max((abs(value) for value in x_value), default=0.0)
    sum_abs_x = sum(abs(value) for value in x_value)
    if max_abs_x > 0.0:
        domain_scale = min(domain_scale, m_bound / max_abs_x)
    if sum_abs_x > 0.0:
        domain_scale = min(domain_scale, k * m_bound / sum_abs_x)
    for row_value, rhs in zip(matvec(a, x_value), b):
        if row_value > rhs and row_value > 0.0:
            domain_scale = min(domain_scale, rhs / row_value)
    if domain_scale < 1.0:
        x_value = [domain_scale * value for value in x_value]
    external_objective = primal_objective(x_value, v, gamma, k, m_bound)
    violation = max_linear_violation(a, b, x_value)
    elapsed = time.perf_counter() - start
    return SolverResult(
        "Gurobi",
        x_value,
        None,
        external_objective,
        None,
        None,
        violation,
        int(model.BarIterCount),
        0,
        elapsed,
        "optimal" if model.Status == GRB.OPTIMAL else "suboptimal",
        [
            TracePoint(
                iteration=int(model.BarIterCount),
                pava_calls=0,
                seconds=elapsed,
                objective=external_objective,
                dual_value=external_objective,
                certificate_gap=0.0,
                violation=violation,
                kkt_residual=0.0,
            )
        ],
    )


def large_benchmark_data(
    d: int = 2000,
    m: int = 50,
    k: Optional[int] = None,
    gamma: float = 0.75,
    m_bound: float = 2.0,
    row_nonzeros: int = 64,
    seed: int = 7,
) -> Tuple[Matrix, Vector, Vector, float, int, float]:
    """Generate a reproducible, nontrivial large constrained-prox instance.

    Every row is normalized, sparse, and oriented so that it cuts off the
    unconstrained proximal point.  Its right-hand side is a fixed fraction of
    that point's row value.  Therefore x = 0 is strictly feasible, while the
    unconstrained solution violates every generated inequality.
    """
    if d < 2:
        raise ValueError("d must be at least 2")
    if m < 1:
        raise ValueError("m must be positive")
    if k is None:
        k = max(1, d // 5)
    if not 1 <= k <= d:
        raise ValueError("k must be in {1, ..., d}")
    if row_nonzeros < 1:
        raise ValueError("row_nonzeros must be positive")

    v = [
        (-1.0 if i % 2 else 1.0) * (2.8 - 1.8 * i / (d - 1) + 0.10 * math.sin(0.17 * i))
        for i in range(d)
    ]
    unconstrained = perspective_prox_state(v, gamma, k, m_bound).x
    active_coordinates = [
        i for i, value in enumerate(unconstrained) if abs(value) > 1e-9
    ]
    if len(active_coordinates) < min(m, row_nonzeros):
        raise ValueError(
            "Too few nonzero coordinates in the unconstrained prox; increase k"
        )

    rng = random.Random(seed)
    support_size = min(row_nonzeros, len(active_coordinates))
    a: Matrix = []
    b: Vector = []
    for row_number in range(m):
        support = rng.sample(active_coordinates, support_size)
        row = [0.0] * d
        for i in support:
            sign = 1.0 if unconstrained[i] >= 0.0 else -1.0
            row[i] = sign * (0.5 + rng.random())
        row_norm = norm2(row)
        row = [coefficient / row_norm for coefficient in row]
        unconstrained_row_value = dot(row, unconstrained)
        cut_ratio = 0.65 + 0.20 * (row_number % 7) / 6.0
        a.append(row)
        b.append(cut_ratio * unconstrained_row_value)
    unconstrained_rows = matvec(a, unconstrained)
    if min(b) <= 0.0 or not all(
        value > rhs for value, rhs in zip(unconstrained_rows, b)
    ):
        raise RuntimeError("The generated benchmark failed its construction check")
    return a, b, v, gamma, k, m_bound


def run_pava_checks() -> None:
    tolerances: List[Tuple[str, float]] = []

    u = [2.0, -1.0, 0.25]
    h = [0.3, -0.2, 0.4]
    gamma = 0.7
    state = perspective_prox_state(u, gamma, len(u), 100.0)
    expected_x = [ui / (1.0 + gamma) for ui in u]
    expected_jh = [hi / (1.0 + gamma) for hi in h]
    tolerances.append(
        (
            "all-selected prox",
            norm_inf([xi - ei for xi, ei in zip(state.x, expected_x)]),
        )
    )
    tolerances.append(
        (
            "all-selected JVP",
            norm_inf([ji - ei for ji, ei in zip(state.jvp(h), expected_jh)]),
        )
    )

    scalar_inside = perspective_prox_state([1.0], 0.7, 1, 2.0)
    scalar_outside = perspective_prox_state([5.0], 0.7, 1, 2.0)
    tolerances.append(("scalar inside", abs(scalar_inside.x[0] - 1.0 / 1.7)))
    tolerances.append(("scalar outside", abs(scalar_outside.x[0] - 2.0)))

    pooled_u = [6.0, 4.0, 3.5, 1.0]
    pooled_h = [0.2, -0.3, 0.4, 0.1]
    pooled = perspective_prox_state(pooled_u, 1.0, 2, 100.0)
    epsilon = 1e-6
    plus = perspective_prox_state(
        add_scaled(pooled_u, epsilon, pooled_h), 1.0, 2, 100.0
    )
    minus = perspective_prox_state(
        add_scaled(pooled_u, -epsilon, pooled_h), 1.0, 2, 100.0
    )
    finite_difference = [
        (plus_i - minus_i) / (2.0 * epsilon) for plus_i, minus_i in zip(plus.x, minus.x)
    ]
    tolerances.append(
        (
            "pooled finite-difference JVP",
            norm_inf(
                [fi - ji for fi, ji in zip(finite_difference, pooled.jvp(pooled_h))]
            ),
        )
    )

    rng = random.Random(19)
    topk_prox_error = 0.0
    topk_jvp_error = 0.0
    topk_envelope_error = 0.0
    for dimension in (3, 10, 50):
        for selected in (1, max(1, dimension // 5), dimension):
            for _ in range(8):
                test_u = [rng.uniform(-5.0, 5.0) for _ in range(dimension)]
                test_h = [rng.uniform(-1.0, 1.0) for _ in range(dimension)]
                test_gamma = rng.uniform(0.2, 3.0)
                test_bound = rng.uniform(0.5, 4.0)
                full = perspective_prox_state(test_u, test_gamma, selected, test_bound)
                partial = perspective_prox_state_topk(
                    test_u, test_gamma, selected, test_bound
                )
                topk_prox_error = max(
                    topk_prox_error,
                    norm_inf([xi - yi for xi, yi in zip(full.x, partial.x)]),
                )
                topk_jvp_error = max(
                    topk_jvp_error,
                    norm_inf(
                        [
                            xi - yi
                            for xi, yi in zip(full.jvp(test_h), partial.jvp(test_h))
                        ]
                    ),
                )
                topk_envelope_error = max(
                    topk_envelope_error,
                    abs(full.moreau_value - partial.moreau_value),
                )
    edge_cases = [
        ([0.0] * 20, 0.75, 10, 2.0),
        ([1.0] * 40, 1.0, 20, 100.0),
        ([3.0, 3.0], 1.0, 1, 2.0),
        ([4.0, -3.0, 2.0, -1.0], 0.5, 4, 1.5),
    ]
    for test_u, test_gamma, selected, test_bound in edge_cases:
        test_h = [((i % 5) - 2.0) / 3.0 for i in range(len(test_u))]
        full = perspective_prox_state(test_u, test_gamma, selected, test_bound)
        partial = perspective_prox_state_topk(test_u, test_gamma, selected, test_bound)
        topk_prox_error = max(
            topk_prox_error,
            norm_inf([xi - yi for xi, yi in zip(full.x, partial.x)]),
        )
        topk_jvp_error = max(
            topk_jvp_error,
            norm_inf(
                [xi - yi for xi, yi in zip(full.jvp(test_h), partial.jvp(test_h))]
            ),
        )
        topk_envelope_error = max(
            topk_envelope_error,
            abs(full.moreau_value - partial.moreau_value),
        )
    tolerances.append(("top-k prox equivalence", topk_prox_error))
    tolerances.append(("top-k JVP equivalence", topk_jvp_error))
    tolerances.append(("top-k envelope equivalence", topk_envelope_error))

    worst = max(value for _, value in tolerances)
    if worst > 1e-7:
        details = ", ".join(f"{name}={value:.3e}" for name, value in tolerances)
        raise AssertionError(f"PAVA self-check failed: {details}")


def run_pava_speed_benchmark(
    dimension: int = 100000,
    selected: int = 100,
    repeats: int = 5,
    seed: int = 31,
) -> Tuple[float, float]:
    if not 1 <= selected <= dimension:
        raise ValueError("The PAVA benchmark requires 1 <= k <= d")
    rng = random.Random(seed)
    u = [rng.uniform(-4.0, 4.0) for _ in range(dimension)]
    gamma = 0.75
    m_bound = 2.0

    full = perspective_prox_state(u, gamma, selected, m_bound)
    partial = perspective_prox_state_topk(u, gamma, selected, m_bound)
    mismatch = norm_inf([xi - yi for xi, yi in zip(full.x, partial.x)])
    if mismatch > 1e-9:
        raise AssertionError(
            f"Top-k PAVA disagrees with full sorting by {mismatch:.3e}"
        )

    full_times: Vector = []
    topk_times: Vector = []
    for _ in range(repeats):
        start = time.perf_counter()
        perspective_prox_state(u, gamma, selected, m_bound)
        full_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        perspective_prox_state_topk(u, gamma, selected, m_bound)
        topk_times.append(time.perf_counter() - start)

    full_seconds = sorted(full_times)[len(full_times) // 2]
    topk_seconds = sorted(topk_times)[len(topk_times) // 2]
    print(
        f"PAVA-only benchmark: d = {dimension}, k = {selected}, "
        f"max difference = {mismatch:.3e}"
    )
    print(
        f"  full sort = {1e3 * full_seconds:.3f} ms, "
        f"top-k only = {1e3 * topk_seconds:.3f} ms, "
        f"speedup = {full_seconds / topk_seconds:.2f}x"
    )
    return full_seconds, topk_seconds


def _svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _finite_bounds(values: Sequence[float]) -> Tuple[float, float]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return 0.0, 1.0
    lower = min(finite)
    upper = max(finite)
    if lower == upper:
        scale = max(1.0, abs(lower))
        return lower - 0.1 * scale, upper + 0.1 * scale
    padding = 0.05 * (upper - lower)
    return lower - padding, upper + padding


def _positive_bounds(values: Sequence[float]) -> Tuple[float, float]:
    positive = [value for value in values if math.isfinite(value) and value > 0.0]
    if not positive:
        return 1e-16, 1.0
    lower = min(positive)
    upper = max(positive)
    if lower == upper:
        return lower / 10.0, upper * 10.0
    return lower, upper


def _polyline_points(
    x_values: Sequence[float],
    y_values: Sequence[float],
    left: float,
    top: float,
    width: float,
    height: float,
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    log_y: bool,
) -> str:
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    if x_max == x_min:
        x_max = x_min + 1.0
    if log_y:
        y_min = math.log10(max(y_min, 1e-300))
        y_max = math.log10(max(y_max, 1e-300))
        if y_max == y_min:
            y_max = y_min + 1.0
    elif y_max == y_min:
        y_max = y_min + 1.0

    points: List[str] = []
    for x_value, y_value in zip(x_values, y_values):
        if not (math.isfinite(x_value) and math.isfinite(y_value)):
            continue
        if log_y:
            if y_value <= 0.0:
                continue
            y_plot = math.log10(y_value)
        else:
            y_plot = y_value
        x_pixel = left + width * (x_value - x_min) / (x_max - x_min)
        y_pixel = top + height * (1.0 - (y_plot - y_min) / (y_max - y_min))
        points.append(f"{x_pixel:.2f},{y_pixel:.2f}")
    return " ".join(points)


def _write_trace_plot_svg(
    results: Sequence[SolverResult],
    output_path: Path,
    *,
    x_key: str,
    y_key: str,
    title: str,
    x_label: str,
    y_label: str,
    log_y: bool = False,
) -> bool:
    series: List[Tuple[str, str, List[float], List[float]]] = []
    palette = ["#0b6e4f", "#c84c09", "#2458b3", "#8a2be2", "#8f1d21"]
    for index, result in enumerate(results):
        trace = result.trace or []
        x_values: List[float] = []
        y_values: List[float] = []
        for point in trace:
            x_values.append(float(getattr(point, x_key)))
            y_values.append(float(getattr(point, y_key)))
        if x_values and y_values:
            series.append((result.method, palette[index % len(palette)], x_values, y_values))
    if not series:
        return False

    all_x = [value for _, _, x_values, _ in series for value in x_values]
    all_y = [value for _, _, _, y_values in series for value in y_values]
    x_bounds = _finite_bounds(all_x)
    if all_x and min(all_x) >= 0.0:
        x_bounds = (0.0, x_bounds[1])
    y_bounds = _positive_bounds(all_y) if log_y else _finite_bounds(all_y)

    width = 960
    height = 640
    left = 90
    right = 36
    top = 56
    bottom = 84
    plot_width = width - left - right
    plot_height = height - top - bottom
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect x="0" y="0" width="100%" height="100%" fill="#fffdf7"/>',
        f'<text x="{left}" y="30" font-size="24" font-family="Segoe UI, sans-serif" '
        f'fill="#1f1f1f">{_svg_escape(title)}</text>',
        (
            f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" '
            'fill="#ffffff" stroke="#d9d4c7"/>'
        ),
    ]

    for tick in range(6):
        ratio = tick / 5.0
        x_pixel = left + ratio * plot_width
        y_pixel = top + ratio * plot_height
        x_value = x_bounds[0] + ratio * (x_bounds[1] - x_bounds[0])
        lines.append(
            f'<line x1="{x_pixel:.2f}" y1="{top}" x2="{x_pixel:.2f}" '
            f'y2="{top + plot_height}" stroke="#eee8da"/>'
        )
        lines.append(
            f'<text x="{x_pixel:.2f}" y="{top + plot_height + 26}" text-anchor="middle" '
            f'font-size="13" font-family="Consolas, monospace" fill="#545454">'
            f"{x_value:.2f}</text>"
        )
        lines.append(
            f'<line x1="{left}" y1="{y_pixel:.2f}" x2="{left + plot_width}" '
            f'y2="{y_pixel:.2f}" stroke="#eee8da"/>'
        )
        if log_y:
            log_lower = math.log10(max(y_bounds[0], 1e-300))
            log_upper = math.log10(max(y_bounds[1], 1e-300))
            y_value = 10.0 ** (log_upper - ratio * (log_upper - log_lower))
            y_label_value = f"{y_value:.1e}"
        else:
            y_value = y_bounds[1] - ratio * (y_bounds[1] - y_bounds[0])
            y_label_value = f"{y_value:.2e}" if abs(y_value) < 1e-2 else f"{y_value:.4f}"
        lines.append(
            f'<text x="{left - 12}" y="{y_pixel + 4:.2f}" text-anchor="end" '
            f'font-size="13" font-family="Consolas, monospace" fill="#545454">'
            f"{y_label_value}</text>"
        )

    for label, color, x_values, y_values in series:
        points = _polyline_points(
            x_values,
            y_values,
            left,
            top,
            plot_width,
            plot_height,
            x_bounds,
            y_bounds,
            log_y,
        )
        if points:
            lines.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="3" '
                f'stroke-linejoin="round" stroke-linecap="round" points="{points}"/>'
            )
        if len(x_values) == 1 and len(y_values) == 1:
            point = _polyline_points(
                x_values,
                y_values,
                left,
                top,
                plot_width,
                plot_height,
                x_bounds,
                y_bounds,
                log_y,
            )
            if point:
                x_pixel, y_pixel = point.split(" ")[0].split(",")
                lines.append(
                    f'<circle cx="{x_pixel}" cy="{y_pixel}" r="4" fill="{color}"/>'
                )

    legend_x = left + 16
    legend_y = top + 18
    for index, (label, color, _, _) in enumerate(series):
        offset = 24 * index
        lines.append(
            f'<line x1="{legend_x}" y1="{legend_y + offset}" '
            f'x2="{legend_x + 28}" y2="{legend_y + offset}" stroke="{color}" '
            'stroke-width="4" stroke-linecap="round"/>'
        )
        lines.append(
            f'<text x="{legend_x + 36}" y="{legend_y + offset + 4}" font-size="14" '
            f'font-family="Segoe UI, sans-serif" fill="#1f1f1f">'
            f"{_svg_escape(label)}</text>"
        )

    lines.extend(
        [
            (
                f'<text x="{left + plot_width / 2:.2f}" y="{height - 24}" '
                'text-anchor="middle" font-size="16" '
                'font-family="Segoe UI, sans-serif" fill="#333333">'
                f"{_svg_escape(x_label)}</text>"
            ),
            (
                f'<text x="28" y="{top + plot_height / 2:.2f}" text-anchor="middle" '
                'font-size="16" font-family="Segoe UI, sans-serif" fill="#333333" '
                f'transform="rotate(-90 28 {top + plot_height / 2:.2f})">'
                f"{_svg_escape(y_label)}</text>"
            ),
            "</svg>",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return True


def _write_solution_plot_svg(
    results: Sequence[SolverResult],
    reference: SolverResult,
    output_path: Path,
) -> None:
    order = sorted(range(len(reference.x)), key=lambda i: (-abs(reference.x[i]), i))
    limit = min(200, len(order))
    chosen = order[:limit]
    x_axis = [float(index + 1) for index in range(limit)]
    palette = ["#2458b3", "#0b6e4f", "#c84c09", "#8a2be2", "#8f1d21"]
    series: List[Tuple[str, str, List[float]]] = []
    plotted_results = [reference] + [
        result for result in results if result.method != reference.method
    ]
    for index, result in enumerate(plotted_results):
        values = [result.x[i] for i in chosen]
        series.append((result.method, palette[index % len(palette)], values))

    all_y = [value for _, _, values in series for value in values]
    x_bounds = _finite_bounds(x_axis)
    if x_axis:
        x_bounds = (0.0, x_bounds[1])
    y_bounds = _finite_bounds(all_y)
    width = 960
    height = 640
    left = 90
    right = 36
    top = 56
    bottom = 84
    plot_width = width - left - right
    plot_height = height - top - bottom
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect x="0" y="0" width="100%" height="100%" fill="#fffdf7"/>',
        (
            f'<text x="{left}" y="30" font-size="24" font-family="Segoe UI, sans-serif" '
            'fill="#1f1f1f">Solution comparison on top |x_G| coordinates</text>'
        ),
        (
            f'<text x="{left}" y="50" font-size="14" font-family="Segoe UI, sans-serif" '
            f'fill="#545454">Showing the largest {limit} coordinates ranked by '
            '|Gurobi solution|.</text>'
        ),
        (
            f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" '
            'fill="#ffffff" stroke="#d9d4c7"/>'
        ),
    ]

    for tick in range(6):
        ratio = tick / 5.0
        x_pixel = left + ratio * plot_width
        y_pixel = top + ratio * plot_height
        x_value = x_bounds[0] + ratio * (x_bounds[1] - x_bounds[0])
        y_value = y_bounds[1] - ratio * (y_bounds[1] - y_bounds[0])
        lines.append(
            f'<line x1="{x_pixel:.2f}" y1="{top}" x2="{x_pixel:.2f}" '
            f'y2="{top + plot_height}" stroke="#eee8da"/>'
        )
        lines.append(
            f'<text x="{x_pixel:.2f}" y="{top + plot_height + 26}" text-anchor="middle" '
            f'font-size="13" font-family="Consolas, monospace" fill="#545454">'
            f"{x_value:.0f}</text>"
        )
        lines.append(
            f'<line x1="{left}" y1="{y_pixel:.2f}" x2="{left + plot_width}" '
            f'y2="{y_pixel:.2f}" stroke="#eee8da"/>'
        )
        lines.append(
            f'<text x="{left - 12}" y="{y_pixel + 4:.2f}" text-anchor="end" '
            f'font-size="13" font-family="Consolas, monospace" fill="#545454">'
            f"{y_value:.3f}</text>"
        )

    for label, color, values in series:
        points = _polyline_points(
            x_axis,
            values,
            left,
            top,
            plot_width,
            plot_height,
            x_bounds,
            y_bounds,
            False,
        )
        if points:
            lines.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="2.5" '
                f'stroke-linejoin="round" stroke-linecap="round" points="{points}"/>'
            )

    legend_x = left + 16
    legend_y = top + 18
    for index, (label, color, _) in enumerate(series):
        offset = 24 * index
        lines.append(
            f'<line x1="{legend_x}" y1="{legend_y + offset}" '
            f'x2="{legend_x + 28}" y2="{legend_y + offset}" stroke="{color}" '
            'stroke-width="4" stroke-linecap="round"/>'
        )
        lines.append(
            f'<text x="{legend_x + 36}" y="{legend_y + offset + 4}" font-size="14" '
            f'font-family="Segoe UI, sans-serif" fill="#1f1f1f">'
            f"{_svg_escape(label)}</text>"
        )

    lines.extend(
        [
            (
                f'<text x="{left + plot_width / 2:.2f}" y="{height - 24}" '
                'text-anchor="middle" font-size="16" '
                'font-family="Segoe UI, sans-serif" fill="#333333">'
                "ranked coordinate index</text>"
            ),
            (
                f'<text x="28" y="{top + plot_height / 2:.2f}" text-anchor="middle" '
                'font-size="16" font-family="Segoe UI, sans-serif" fill="#333333" '
                f'transform="rotate(-90 28 {top + plot_height / 2:.2f})">x value</text>'
            ),
            "</svg>",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_comparison_artifacts(
    results: Sequence[SolverResult],
    reference: SolverResult,
    output_dir: Path,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    summary_path = output_dir / "comparison_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "status",
                "time_ms",
                "iterations",
                "pava_calls",
                "objective",
                "certificate_gap",
                "kkt_residual",
                "linear_violation",
                "distance_to_gurobi_inf",
                "relative_objective_error",
            ]
        )
        for result in results:
            distance = norm_inf([xi - xg for xi, xg in zip(result.x, reference.x)])
            objective_difference = abs(result.objective - reference.objective)
            relative_objective_difference = objective_difference / max(
                1.0, abs(reference.objective)
            )
            writer.writerow(
                [
                    result.method,
                    result.status,
                    1e3 * result.seconds,
                    result.iterations,
                    result.pava_calls,
                    result.objective,
                    "" if result.certificate_gap is None else result.certificate_gap,
                    "" if result.kkt_residual is None else result.kkt_residual,
                    result.violation,
                    distance,
                    relative_objective_difference,
                ]
            )
    written.append(summary_path)

    if _write_trace_plot_svg(
        results,
        output_dir / "comparison_gap_vs_pava.svg",
        x_key="pava_calls",
        y_key="certificate_gap",
        title="Certificate gap vs PAVA calls",
        x_label="PAVA calls",
        y_label="certificate gap",
        log_y=True,
    ):
        written.append(output_dir / "comparison_gap_vs_pava.svg")

    if _write_trace_plot_svg(
        results,
        output_dir / "comparison_gap_vs_time.svg",
        x_key="seconds",
        y_key="certificate_gap",
        title="Certificate gap vs wall-clock time",
        x_label="seconds",
        y_label="certificate gap",
        log_y=True,
    ):
        written.append(output_dir / "comparison_gap_vs_time.svg")

    if _write_trace_plot_svg(
        results,
        output_dir / "comparison_kkt_vs_pava.svg",
        x_key="pava_calls",
        y_key="kkt_residual",
        title="Projected KKT residual vs PAVA calls",
        x_label="PAVA calls",
        y_label="projected KKT residual",
        log_y=True,
    ):
        written.append(output_dir / "comparison_kkt_vs_pava.svg")

    _write_solution_plot_svg(results, reference, output_dir / "comparison_solution.svg")
    written.append(output_dir / "comparison_solution.svg")
    return written


def print_comparison(results: Sequence[SolverResult], reference: SolverResult) -> None:
    print("\nCommon external evaluation")
    print(
        f"{'method':<20} {'status':<15} {'time (ms)':>10} {'iter':>7} "
        f"{'PAVA':>7} {'objective':>16} {'cert. gap':>12} "
        f"{'lin. viol.':>12} {'||x-xG||inf':>13} {'rel. obj.':>12}"
    )
    for result in results:
        distance = norm_inf([xi - xg for xi, xg in zip(result.x, reference.x)])
        objective_difference = abs(result.objective - reference.objective)
        relative_objective_difference = objective_difference / max(
            1.0, abs(reference.objective)
        )
        gap = "-" if result.certificate_gap is None else f"{result.certificate_gap:.3e}"
        print(
            f"{result.method:<20} {result.status:<15} {1e3 * result.seconds:10.3f} "
            f"{result.iterations:7d} {result.pava_calls:7d} {result.objective:16.10f} "
            f"{gap:>12} {result.violation:12.3e} {distance:13.3e} "
            f"{relative_objective_difference:12.3e}"
        )

    print("\nMultiplier/KKT diagnostics")
    for result in results:
        if result.multiplier is not None:
            positive_multipliers = [
                value for value in result.multiplier if value > 1e-8
            ]
            multiplier_summary = (
                f"{len(positive_multipliers)}/{len(result.multiplier)} positive"
            )
            if len(result.multiplier) <= 10:
                multiplier_summary += (
                    f", lambda = {[round(value, 10) for value in result.multiplier]}"
                )
            print(
                f"{result.method:<20} projected residual = {result.kkt_residual:.3e}, "
                f"{multiplier_summary}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Large constrained-prox comparison: FISTA, Newton, and Gurobi"
    )
    parser.add_argument("--d", type=int, default=2000)
    parser.add_argument("--m", type=int, default=100)
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="perspective budget; default is floor(d / 5)",
    )
    parser.add_argument("--row-nonzeros", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--m-bound", type=float, default=2.0)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--restart", type=int, default=25)
    parser.add_argument("--max-fista", type=int, default=20000)
    parser.add_argument("--max-newton", type=int, default=100)
    parser.add_argument(
        "--pava",
        choices=("auto", "full", "topk"),
        default="auto",
        help="base proximal oracle used inside FISTA and Newton",
    )
    parser.add_argument("--benchmark-pava", action="store_true")
    parser.add_argument("--pava-only", action="store_true")
    parser.add_argument("--pava-benchmark-d", type=int, default=100000)
    parser.add_argument("--pava-benchmark-k", type=int, default=100)
    parser.add_argument("--pava-benchmark-repeats", type=int, default=5)
    parser.add_argument("--gurobi-tolerance", type=float, default=1e-9)
    parser.add_argument("--gurobi-log", action="store_true")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable iteration progress bars for the custom solvers",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).with_name("results"),
        help="directory used for CSV/SVG comparison artifacts",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="skip writing comparison artifacts",
    )
    args = parser.parse_args()

    run_pava_checks()
    if args.benchmark_pava or args.pava_only:
        run_pava_speed_benchmark(
            dimension=args.pava_benchmark_d,
            selected=args.pava_benchmark_k,
            repeats=args.pava_benchmark_repeats,
        )
    if args.pava_only:
        return
    a, b, v, gamma, k, m_bound = large_benchmark_data(
        d=args.d,
        m=args.m,
        k=args.k,
        gamma=args.gamma,
        m_bound=args.m_bound,
        row_nonzeros=args.row_nonzeros,
        seed=args.seed,
    )
    nonzeros_a = sum(coefficient != 0.0 for row in a for coefficient in row)
    print(
        f"Instance: d = {len(v)}, m = {len(b)}, k = {k}, M = {m_bound:g}, "
        f"gamma = {gamma:g}, nnz(A) = {nonzeros_a}"
    )
    resolved_pava = "topk" if args.pava == "auto" and 8 * k < len(v) else args.pava
    if resolved_pava == "auto":
        resolved_pava = "full"
    print(f"PAVA oracle: requested = {args.pava}, used = {resolved_pava}")
    print(
        "Construction check: x = 0 is strictly feasible and the "
        "unconstrained prox violates every row."
    )
    lipschitz = gamma * spectral_norm_squared_upper_bound(a)
    fista = restarted_dual_fista(
        a,
        b,
        v,
        gamma,
        k,
        m_bound,
        tolerance=args.tolerance,
        restart_length=args.restart,
        max_iterations=args.max_fista,
        lipschitz=lipschitz,
        pava_method=args.pava,
        show_progress=not args.no_progress,
    )
    newton = globalized_semismooth_newton(
        a,
        b,
        v,
        gamma,
        k,
        m_bound,
        tolerance=args.tolerance,
        max_iterations=args.max_newton,
        lipschitz=lipschitz,
        pava_method=args.pava,
        show_progress=not args.no_progress,
    )
    gurobi = solve_with_gurobi(
        a,
        b,
        v,
        gamma,
        k,
        m_bound,
        output_flag=int(args.gurobi_log),
        barrier_tolerance=args.gurobi_tolerance,
    )
    print_comparison([fista, newton, gurobi], gurobi)
    if not args.no_visualize:
        written = write_comparison_artifacts(
            [fista, newton, gurobi], gurobi, args.results_dir
        )
        print("\nWrote comparison artifacts:")
        for path in written:
            print(f"  {path}")

    accuracy_ok = all(
        norm_inf([xi - xg for xi, xg in zip(result.x, gurobi.x)]) <= 5e-4
        and abs(result.objective - gurobi.objective) / max(1.0, abs(gurobi.objective))
        <= 1e-8
        and result.violation <= 1e-7
        for result in (fista, newton)
    )
    print(f"\nMATCH CHECK: {'PASS' if accuracy_ok else 'FAIL'}")
    if not accuracy_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
