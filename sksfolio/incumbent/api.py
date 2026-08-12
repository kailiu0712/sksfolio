"""Public incumbent search for constrained sparse Markowitz portfolios."""

from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np

from ..relaxation.problem import MarkowitzInstance, load_instance_bundle
from .evaluation import evaluate_incumbent, sparse_objective
from .restricted_qp import RestrictedQPResult, solve_restricted_qp
from .result import IncumbentResult
from .state import IncumbentState
from .support import (
    binary_perspective_prox,
    dependent_round_support,
    perspective_activations,
    rescale_marginals,
    top_k_indices,
    validate_branch_indices,
)


INCUMBENT_METHODS = (
    "auto",
    "fast",
    "quality",
    "topk",
    "binary_prox",
    "randomized",
    "prune",
    "discrete_first_order",
    "swap",
)
DEFAULT_INCUMBENT_METHOD = "auto"
DEFAULT_RESTRICTED_SOLVER = "osqp"
ProblemLike = Union[MarkowitzInstance, str, Path]


def _problem(value: ProblemLike) -> MarkowitzInstance:
    if isinstance(value, MarkowitzInstance):
        value.validate()
        return value
    return load_instance_bundle(value)


def _raw_mapping(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    raw = getattr(value, "raw", value)
    return raw if isinstance(raw, Mapping) else {}


def _weights(value: Any, dimension: int) -> Optional[np.ndarray]:
    if value is None:
        return None
    candidate = getattr(value, "weights", None)
    if candidate is None:
        raw = _raw_mapping(value)
        candidate = raw.get("x")
    if candidate is None and not isinstance(value, Mapping):
        try:
            candidate = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            return None
    if candidate is None:
        return None
    vector = np.asarray(candidate, dtype=float).reshape(-1)
    if vector.shape != (dimension,):
        raise ValueError("supplied solution has the wrong dimension")
    return vector


def _relaxation_lower_bound(value: Any) -> Optional[float]:
    if value is None:
        return None
    candidate = getattr(value, "safe_dual_bound", None)
    if candidate is None:
        raw = _raw_mapping(value)
        candidate = raw.get(
            "best_dual_lower_bound",
            raw.get("dual_bound"),
        )
    try:
        result = float(candidate)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _relaxation_constraint_dual(
    value: Any,
    rows: int,
) -> Optional[np.ndarray]:
    raw = _raw_mapping(value)
    candidate = raw.get("current_constraint_dual")
    if candidate is None:
        return None
    array = np.asarray(candidate, dtype=float).reshape(-1)
    if array.shape != (rows,) or np.any(~np.isfinite(array)):
        return None
    return array


def _normalized(values: np.ndarray) -> np.ndarray:
    result = np.maximum(np.asarray(values, dtype=float), 0.0)
    maximum = float(np.max(result)) if result.size else 0.0
    return result / maximum if maximum > 0.0 else result


class _IncumbentSearch:
    def __init__(
        self,
        instance: MarkowitzInstance,
        reference_x: np.ndarray,
        relaxation: Any,
        restricted_solver: str,
        required: np.ndarray,
        forbidden: np.ndarray,
        settings: Mapping[str, Any],
    ) -> None:
        self.instance = instance
        self.reference_x = np.clip(reference_x, 0.0, 1.0)
        self.activations = perspective_activations(
            self.reference_x,
            instance.k,
        )
        self.relaxation_dual = _relaxation_constraint_dual(
            relaxation,
            instance.rows,
        )
        self.lower_bound = _relaxation_lower_bound(relaxation)
        self.restricted_solver = restricted_solver
        self.required = required
        self.forbidden = forbidden
        self.settings = dict(settings)
        self.start = time.perf_counter()
        limit = self.settings.get("time_limit")
        self.deadline = (
            math.inf if limit is None else self.start + max(float(limit), 0.0)
        )
        self.rng = np.random.default_rng(
            int(self.settings.get("random_state", 0))
        )
        self.cache: dict[tuple[int, ...], RestrictedQPResult] = {}
        self.history: list[dict[str, Any]] = []
        self.best_x: Optional[np.ndarray] = None
        self.best_selectors: Optional[np.ndarray] = None
        self.best_objective = math.inf
        self.best_result: Optional[RestrictedQPResult] = None
        self.best_method: Optional[str] = None
        self.qp_seconds = 0.0
        self.feasible_candidates = 0
        self.sample_counter = 0
        self.binary_gains: Optional[np.ndarray] = None

    def remaining(self) -> Optional[float]:
        if not math.isfinite(self.deadline):
            return None
        return max(0.0, self.deadline - time.perf_counter())

    def expired(self) -> bool:
        remaining = self.remaining()
        return remaining is not None and remaining <= 0.0

    def normalize_support(
        self,
        support: Sequence[int],
        *,
        allow_large: bool = False,
    ) -> np.ndarray:
        selected = np.unique(np.asarray(support, dtype=np.int64).reshape(-1))
        if np.any(selected < 0) or np.any(selected >= self.instance.dimension):
            raise ValueError("candidate support contains an invalid index")
        selected = np.union1d(selected, self.required)
        if self.forbidden.size:
            selected = np.setdiff1d(
                selected,
                self.forbidden,
                assume_unique=True,
            )
        if not allow_large and selected.size > self.instance.k:
            score = np.zeros(self.instance.dimension)
            score[selected] = self.activations[selected]
            selected = top_k_indices(
                score,
                self.instance.k,
                required=self.required,
                forbidden=self.forbidden,
            )
        return selected.astype(np.int64, copy=False)

    def install_warm(self, warm_start: Any) -> None:
        vector = _weights(warm_start, self.instance.dimension)
        if vector is None:
            return
        raw = _raw_mapping(warm_start)
        selectors = raw.get("selectors")
        state = IncumbentState.coerce(warm_start)
        if selectors is None and state is not None:
            selectors = state.selectors
        if selectors is None:
            selectors = np.abs(vector) > float(
                self.settings.get("active_tolerance", 1e-9)
            )
        diagnostics = evaluate_incumbent(
            self.instance,
            vector,
            selectors,
            feasibility_tolerance=float(
                self.settings.get("feasibility_tolerance", 1e-7)
            ),
            active_tolerance=float(
                self.settings.get("active_tolerance", 1e-9)
            ),
            required_assets=self.required,
            forbidden_assets=self.forbidden,
        )
        if diagnostics["numerically_feasible"]:
            self.best_x = vector.copy()
            self.best_selectors = diagnostics["selectors"].copy()
            self.best_objective = float(diagnostics["objective"])
            self.best_method = "warm_start"
            self.history.append(
                {
                    "method": "warm_start",
                    "support_size": int(np.sum(self.best_selectors)),
                    "active_size": int(diagnostics["cardinality"]),
                    "feasible": True,
                    "objective": self.best_objective,
                    "incumbent": True,
                    "elapsed_seconds": time.perf_counter() - self.start,
                    "cached": False,
                }
            )
        support = np.flatnonzero(np.asarray(selectors).reshape(-1) > 0.5)
        if support.size <= self.instance.k and not self.expired():
            self.solve_support(
                support,
                "warm_refit",
                warm_start=vector,
            )

    def solve_support(
        self,
        support: Sequence[int],
        method: str,
        *,
        warm_start: Optional[np.ndarray] = None,
        allow_large: bool = False,
    ) -> Optional[RestrictedQPResult]:
        selected = self.normalize_support(support, allow_large=allow_large)
        key = tuple(int(value) for value in selected)
        cached = key in self.cache
        if cached:
            result = self.cache[key]
        else:
            if self.expired():
                return None
            qp_options = dict(self.settings.get("restricted_options", {}))
            qp_options.setdefault(
                "feasibility_tolerance",
                float(self.settings.get("feasibility_tolerance", 1e-7)),
            )
            qp_options.setdefault(
                "active_tolerance",
                float(self.settings.get("active_tolerance", 1e-9)),
            )
            remaining = self.remaining()
            if remaining is not None:
                old_limit = qp_options.get("time_limit")
                qp_options["time_limit"] = (
                    remaining
                    if old_limit is None
                    else min(float(old_limit), remaining)
                )
            result = solve_restricted_qp(
                self.instance,
                selected,
                solver=self.restricted_solver,
                warm_start=warm_start,
                options=qp_options,
            )
            # Do not permanently cache an interrupted numerical attempt.
            # A later call can have a better warm start or more time.
            if (
                result.optimality_certified
                or result.infeasibility_certified
            ):
                self.cache[key] = result
            self.qp_seconds += result.solve_seconds

        is_integer_support = selected.size <= self.instance.k
        incumbent = False
        if result.feasible and is_integer_support:
            self.feasible_candidates += 1
            selectors = np.zeros(self.instance.dimension)
            selectors[selected] = 1.0
            diagnostics = evaluate_incumbent(
                self.instance,
                result.x,
                selectors,
                feasibility_tolerance=float(
                    self.settings.get("feasibility_tolerance", 1e-7)
                ),
                active_tolerance=float(
                    self.settings.get("active_tolerance", 1e-9)
                ),
                required_assets=self.required,
                forbidden_assets=self.forbidden,
            )
            if diagnostics["numerically_feasible"]:
                objective = float(diagnostics["objective"])
                if objective < self.best_objective - float(
                    self.settings.get("improvement_tolerance", 1e-11)
                ):
                    self.best_x = result.x.copy()
                    self.best_selectors = selectors
                    self.best_objective = objective
                    self.best_result = result
                    self.best_method = method
                    incumbent = True
        self.history.append(
            {
                "method": method,
                "support_size": int(selected.size),
                "active_size": int(
                    np.count_nonzero(
                        np.abs(result.x)
                        > float(self.settings.get("active_tolerance", 1e-9))
                    )
                ),
                "feasible": bool(result.feasible and is_integer_support),
                "objective": result.objective,
                "incumbent": incumbent,
                "solver": result.solver,
                "solver_status": result.status,
                "solve_seconds": result.solve_seconds,
                "elapsed_seconds": time.perf_counter() - self.start,
                "cached": cached,
            }
        )
        return result

    def topk(self) -> Optional[RestrictedQPResult]:
        support = top_k_indices(
            self.reference_x,
            self.instance.k,
            required=self.required,
            forbidden=self.forbidden,
        )
        return self.solve_support(support, "topk")

    def binary_prox(self) -> Optional[RestrictedQPResult]:
        factor_gram = np.asarray(self.instance.B.T @ self.instance.B)
        lipschitz = max(
            float(np.linalg.eigvalsh(factor_gram)[-1]),
            1e-12,
        )
        step = float(self.settings.get("binary_step", 1.0 / lipschitz))
        gradient = (
            self.instance.B @ (self.instance.B.T @ self.reference_x)
            - float(self.instance.return_reward) * self.instance.mu
        )
        if (
            bool(self.settings.get("use_relaxation_dual", True))
            and self.relaxation_dual is not None
        ):
            gradient = gradient + self.instance.C.T @ self.relaxation_dual
        center = self.reference_x - step * np.asarray(gradient).reshape(-1)
        _, support, gains = binary_perspective_prox(
            center,
            step,
            float(self.instance.perspective_weight),
            self.instance.k,
        )
        self.binary_gains = gains
        support = self.normalize_support(support)
        if support.size < self.instance.k:
            score = gains + 1e-12 * _normalized(self.reference_x)
            support = top_k_indices(
                score,
                self.instance.k,
                required=np.union1d(self.required, support),
                forbidden=self.forbidden,
            )
        return self.solve_support(support, "binary_prox")

    def probabilities(self) -> np.ndarray:
        result = np.zeros(self.instance.dimension)
        allowed = np.ones(self.instance.dimension, dtype=bool)
        allowed[self.forbidden] = False
        allowed[self.required] = False
        result[self.required] = 1.0
        target = min(
            self.instance.k - self.required.size,
            int(np.count_nonzero(allowed)),
        )
        if target > 0:
            result[allowed] = rescale_marginals(
                self.activations[allowed],
                target,
            )
        return result

    def randomized(self, samples: int) -> None:
        probabilities = self.probabilities()
        for sample in range(max(0, int(samples))):
            if self.expired():
                return
            support = dependent_round_support(
                probabilities,
                self.rng,
                int(round(float(np.sum(probabilities)))),
            )
            self.sample_counter += 1
            self.solve_support(support, f"randomized_{sample + 1}")

    def prune(self) -> Optional[RestrictedQPResult]:
        dimension = self.instance.dimension
        multiplier = max(float(self.settings.get("pool_multiplier", 3.0)), 1.0)
        maximum_multiplier = max(
            float(self.settings.get("maximum_pool_multiplier", 8.0)),
            multiplier,
        )
        score = _normalized(self.reference_x) + _normalized(self.activations)
        if self.binary_gains is not None:
            score += _normalized(self.binary_gains)
        if self.best_selectors is not None:
            score += 2.0 * self.best_selectors

        current: Optional[RestrictedQPResult] = None
        current_support: Optional[np.ndarray] = None
        while multiplier <= maximum_multiplier + 1e-12 and not self.expired():
            pool_size = min(
                dimension - self.forbidden.size,
                max(self.instance.k, int(math.ceil(multiplier * self.instance.k))),
            )
            pool = top_k_indices(
                score,
                pool_size,
                required=self.required,
                forbidden=self.forbidden,
            )
            current = self.solve_support(
                pool,
                f"prune_pool_{pool_size}",
                warm_start=self.reference_x,
                allow_large=True,
            )
            if current is not None and current.feasible:
                current_support = pool
                break
            multiplier *= 2.0
        if current is None or not current.feasible or current_support is None:
            return current

        maximum_refits = int(self.settings.get("maximum_prune_refits", 64))
        removal_trials = int(self.settings.get("prune_removal_trials", 6))
        batch_limit = int(self.settings.get("prune_batch_size", 8))
        refits = 0
        while current_support.size > self.instance.k and not self.expired():
            active = np.flatnonzero(
                np.abs(current.x)
                > float(self.settings.get("active_tolerance", 1e-9))
            )
            active = np.intersect1d(active, current_support, assume_unique=True)
            compressed = np.union1d(active, self.required)
            if compressed.size <= self.instance.k:
                return self.solve_support(
                    compressed,
                    "prune_compressed",
                    warm_start=current.x,
                )
            removable = np.setdiff1d(
                current_support,
                self.required,
                assume_unique=True,
            )
            if removable.size == 0 or refits >= maximum_refits:
                break
            exposure = self.instance.B.T @ current.x
            gradient = (
                self.instance.B @ exposure
                + float(self.instance.perspective_weight) * current.x
                - float(self.instance.return_reward) * self.instance.mu
            )
            diagonal = (
                np.sum(np.asarray(self.instance.B) ** 2, axis=1)
                + float(self.instance.perspective_weight)
            )
            deletion = (
                -current.x * np.asarray(gradient).reshape(-1)
                + 0.5 * diagonal * current.x * current.x
            )
            order = removable[
                np.lexsort((removable, deletion[removable]))
            ]
            excess = current_support.size - self.instance.k
            batch = min(batch_limit, excess, order.size)
            accepted: Optional[tuple[np.ndarray, RestrictedQPResult]] = None
            while batch >= 1 and accepted is None:
                trial_count = 1 if batch > 1 else min(removal_trials, order.size)
                for offset in range(trial_count):
                    removed = (
                        order[:batch]
                        if batch > 1
                        else order[offset : offset + 1]
                    )
                    proposed = np.setdiff1d(
                        current_support,
                        removed,
                        assume_unique=True,
                    )
                    candidate = self.solve_support(
                        proposed,
                        "prune_delete",
                        warm_start=current.x,
                        allow_large=proposed.size > self.instance.k,
                    )
                    refits += 1
                    if candidate is not None and candidate.feasible:
                        accepted = (proposed, candidate)
                        break
                    if refits >= maximum_refits or self.expired():
                        break
                batch //= 2
            if accepted is None:
                break
            current_support, current = accepted
        if current_support.size <= self.instance.k:
            return self.solve_support(
                current_support,
                "prune_final",
                warm_start=current.x,
            )
        return current

    def discrete_first_order(self) -> None:
        if self.best_x is None or self.best_selectors is None:
            return
        current_support = np.flatnonzero(self.best_selectors > 0.5)
        current = self.solve_support(
            current_support,
            "dfo_start",
            warm_start=self.best_x,
        )
        if current is None or not current.feasible:
            return
        average_signal = np.zeros(self.instance.dimension)
        lipschitz = max(float(self.settings.get("dfo_lipschitz", 10.0)), 1e-12)
        iterations = int(self.settings.get("dfo_iterations", 8))
        ridge = float(self.instance.perspective_weight)
        for iteration in range(1, iterations + 1):
            if self.expired():
                return
            exposure = self.instance.B.T @ current.x
            priced = (
                float(self.instance.return_reward) * self.instance.mu
                - self.instance.B @ exposure
            )
            if current.constraint_dual is not None:
                priced = priced - self.instance.C.T @ current.constraint_dual
            signal = np.maximum(np.asarray(priced).reshape(-1), 0.0)
            average_signal += (signal - average_signal) / float(iteration)
            score = current.x + average_signal * average_signal / (
                2.0 * ridge * lipschitz
            )
            proposed = top_k_indices(
                score,
                self.instance.k,
                required=self.required,
                forbidden=self.forbidden,
            )
            if np.array_equal(proposed, current_support):
                return
            candidate = self.solve_support(
                proposed,
                f"discrete_first_order_{iteration}",
                warm_start=current.x,
            )
            if candidate is None or not candidate.feasible:
                return
            current_support = proposed
            current = candidate

    def swap(self) -> None:
        if self.best_x is None or self.best_selectors is None:
            return
        rounds = int(self.settings.get("swap_rounds", 2))
        entry_count = int(self.settings.get("swap_entry_candidates", 8))
        exit_count = int(self.settings.get("swap_exit_candidates", 4))
        maximum_evaluations = int(
            self.settings.get("maximum_swap_evaluations", 24)
        )
        evaluations = 0
        for round_index in range(rounds):
            if self.expired() or evaluations >= maximum_evaluations:
                return
            support = np.flatnonzero(self.best_selectors > 0.5)
            current = self.solve_support(
                support,
                f"swap_start_{round_index + 1}",
                warm_start=self.best_x,
            )
            if current is None or not current.feasible:
                return
            exposure = self.instance.B.T @ current.x
            reduced = (
                self.instance.B @ exposure
                + float(self.instance.perspective_weight) * current.x
                - float(self.instance.return_reward) * self.instance.mu
            )
            if current.constraint_dual is not None:
                reduced = reduced + self.instance.C.T @ current.constraint_dual
            reduced = np.asarray(reduced).reshape(-1)
            diagonal = (
                np.sum(np.asarray(self.instance.B) ** 2, axis=1)
                + float(self.instance.perspective_weight)
            )
            entry_score = np.maximum(-reduced, 0.0) ** 2 / np.maximum(
                diagonal,
                1e-14,
            )
            entry_score += float(
                self.settings.get("swap_relaxation_weight", 1e-3)
            ) * _normalized(self.activations)
            excluded = np.union1d(support, self.forbidden)
            entry_score[excluded] = -math.inf
            entrants = top_k_indices(
                entry_score,
                min(entry_count, self.instance.dimension),
                forbidden=excluded,
            )
            removable = np.setdiff1d(
                support,
                self.required,
                assume_unique=True,
            )
            if entrants.size == 0 or removable.size == 0:
                return
            exits = removable[
                np.lexsort((removable, np.abs(current.x[removable])))
            ][:exit_count]
            old_objective = self.best_objective
            best_trial: Optional[RestrictedQPResult] = None
            best_support: Optional[np.ndarray] = None
            best_value = old_objective
            for leaving in exits:
                for entering in entrants:
                    if evaluations >= maximum_evaluations or self.expired():
                        break
                    proposed = support[support != leaving]
                    proposed = np.sort(np.append(proposed, entering))
                    candidate = self.solve_support(
                        proposed,
                        "swap_trial",
                        warm_start=current.x,
                    )
                    evaluations += 1
                    if (
                        candidate is not None
                        and candidate.feasible
                        and candidate.objective is not None
                        and candidate.objective < best_value - float(
                            self.settings.get("improvement_tolerance", 1e-11)
                        )
                    ):
                        best_trial = candidate
                        best_support = proposed
                        best_value = float(candidate.objective)
                if evaluations >= maximum_evaluations or self.expired():
                    break
            if best_trial is None or best_support is None:
                return
            # The trial was already installed by solve_support.  Re-solve is
            # cached and records the accepted semantic step in the history.
            self.solve_support(
                best_support,
                f"swap_accept_{round_index + 1}",
                warm_start=best_trial.x,
            )

    def result(self, requested_method: str) -> IncumbentResult:
        elapsed = time.perf_counter() - self.start
        if self.best_x is None or self.best_selectors is None:
            return IncumbentResult(
                {
                    "status": "time_limit" if self.expired() else "no_feasible_support",
                    "method": requested_method,
                    "x": None,
                    "selectors": None,
                    "upper_bound": None,
                    "numerically_feasible": False,
                    "relaxation_lower_bound": self.lower_bound,
                    "safe_gap": None,
                    "history": self.history,
                    "candidate_count": len(self.cache),
                    "feasible_candidate_count": self.feasible_candidates,
                    "restricted_qp_seconds": self.qp_seconds,
                    "solve_seconds": elapsed,
                    "sample_counter": self.sample_counter,
                    "state": None,
                }
            )
        diagnostics = evaluate_incumbent(
            self.instance,
            self.best_x,
            self.best_selectors,
            feasibility_tolerance=float(
                self.settings.get("feasibility_tolerance", 1e-7)
            ),
            active_tolerance=float(
                self.settings.get("active_tolerance", 1e-9)
            ),
            required_assets=self.required,
            forbidden_assets=self.forbidden,
        )
        upper_bound = float(diagnostics["objective"])
        gap = (
            upper_bound - self.lower_bound
            if self.lower_bound is not None
            else None
        )
        state = IncumbentState(
            dimension=self.instance.dimension,
            k=self.instance.k,
            constraint_ids=tuple(self.instance.constraint_names),
            x=self.best_x.copy(),
            selectors=self.best_selectors.copy(),
            objective=upper_bound,
            required_assets=tuple(int(value) for value in self.required),
            forbidden_assets=tuple(int(value) for value in self.forbidden),
            sample_counter=self.sample_counter,
        )
        return IncumbentResult(
            {
                "status": "feasible",
                "method": requested_method,
                "winning_method": self.best_method,
                "x": self.best_x.copy(),
                "selectors": self.best_selectors.copy(),
                "upper_bound": upper_bound,
                "numerically_feasible": True,
                "floating_point_certified": False,
                "upper_bound_numerically_verified": True,
                "upper_bound_safe_in_exact_arithmetic": False,
                "relaxation_lower_bound": self.lower_bound,
                "safe_gap": gap,
                "relative_safe_gap": (
                    gap / max(1.0, abs(upper_bound))
                    if gap is not None
                    else None
                ),
                "diagnostics": diagnostics,
                "history": self.history,
                "candidate_count": len(self.cache),
                "feasible_candidate_count": self.feasible_candidates,
                "restricted_qp_seconds": self.qp_seconds,
                "solve_seconds": elapsed,
                "sample_counter": self.sample_counter,
                "active_tolerance": float(
                    self.settings.get("active_tolerance", 1e-9)
                ),
                "state": state.to_dict(copy=False),
            }
        )


def solve_incumbent(
    problem: ProblemLike,
    relaxation: Optional[Any] = None,
    *,
    method: str = DEFAULT_INCUMBENT_METHOD,
    restricted_solver: str = DEFAULT_RESTRICTED_SOLVER,
    warm_start: Optional[Any] = None,
    required_assets: Sequence[int] = (),
    forbidden_assets: Sequence[int] = (),
    time_limit: Optional[float] = None,
    random_state: int = 0,
    options: Optional[Mapping[str, Any]] = None,
) -> IncumbentResult:
    """Find a feasible sparse portfolio and a valid numerical upper bound.

    The default ``auto`` policy first tries deterministic top-k and the exact
    unconstrained binary-prox support, then uses dependent rounding and a
    small local-search polish.  Every support is refitted against the full
    original interval rows.  Consequently, an infeasible rounded vector is
    never reported as an incumbent.
    """
    if method not in INCUMBENT_METHODS:
        raise ValueError("method must be one of " + ", ".join(INCUMBENT_METHODS))
    instance = _problem(problem)
    required, forbidden = validate_branch_indices(
        instance.dimension,
        instance.k,
        required_assets,
        forbidden_assets,
    )
    settings = dict(options or {})
    if time_limit is not None:
        if "time_limit" in settings:
            raise ValueError("pass time_limit either directly or in options")
        settings["time_limit"] = float(time_limit)
    settings.setdefault("random_state", int(random_state))
    reference = _weights(relaxation, instance.dimension)
    if reference is None:
        reference = np.asarray(instance.anchor, dtype=float).reshape(-1)
    search = _IncumbentSearch(
        instance,
        reference,
        relaxation,
        restricted_solver,
        required,
        forbidden,
        settings,
    )
    if warm_start is not None:
        search.install_warm(warm_start)

    if method == "topk":
        search.topk()
    elif method == "binary_prox":
        search.binary_prox()
    elif method == "randomized":
        search.randomized(int(settings.get("random_samples", 16)))
    elif method == "prune":
        search.topk()
        search.binary_prox()
        search.prune()
    elif method == "discrete_first_order":
        search.topk()
        search.binary_prox()
        if search.best_x is None:
            search.randomized(int(settings.get("random_samples", 8)))
        search.discrete_first_order()
    elif method == "swap":
        search.topk()
        search.binary_prox()
        if search.best_x is None:
            search.randomized(int(settings.get("random_samples", 8)))
        search.swap()
    elif method == "fast":
        search.topk()
        search.binary_prox()
    else:
        quality = method == "quality"
        search.topk()
        search.binary_prox()
        if (
            quality
            or search.best_x is None
            or bool(settings.get("always_prune", False))
        ):
            search.prune()
        search.randomized(
            int(settings.get("random_samples", 24 if quality else 8))
        )
        search.discrete_first_order()
        if quality:
            search.settings.setdefault("swap_rounds", 4)
            search.settings.setdefault("maximum_swap_evaluations", 64)
        search.swap()
    return search.result(method)


class SparseIncumbent:
    """Estimator-style wrapper matching :class:`PerspectiveRelaxation`."""

    def __init__(
        self,
        method: str = DEFAULT_INCUMBENT_METHOD,
        *,
        restricted_solver: str = DEFAULT_RESTRICTED_SOLVER,
        random_state: int = 0,
        time_limit: Optional[float] = None,
        solver_params: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.method = method
        self.restricted_solver = restricted_solver
        self.random_state = random_state
        self.time_limit = time_limit
        self.solver_params = dict(solver_params or {})

    def fit(
        self,
        problem: ProblemLike,
        relaxation: Optional[Any] = None,
        *,
        warm_start: Optional[Any] = None,
        required_assets: Sequence[int] = (),
        forbidden_assets: Sequence[int] = (),
    ) -> "SparseIncumbent":
        self.result_ = solve_incumbent(
            problem,
            relaxation,
            method=self.method,
            restricted_solver=self.restricted_solver,
            warm_start=warm_start,
            required_assets=required_assets,
            forbidden_assets=forbidden_assets,
            time_limit=self.time_limit,
            random_state=self.random_state,
            options=self.solver_params,
        )
        if not self.result_.feasible:
            raise RuntimeError(
                "incumbent search did not find a feasible sparse portfolio"
            )
        self.weights_ = self.result_.weights.copy()
        self.selectors_ = self.result_.selectors.copy()
        self.upper_bound_ = self.result_.upper_bound
        self.state_ = self.result_.state
        return self


__all__ = [
    "DEFAULT_INCUMBENT_METHOD",
    "DEFAULT_RESTRICTED_SOLVER",
    "INCUMBENT_METHODS",
    "SparseIncumbent",
    "solve_incumbent",
]
