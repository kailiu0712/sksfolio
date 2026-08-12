"""Continuous perspective relaxation for sparse Markowitz portfolios."""

from .api import (
    BACKENDS,
    DEFAULT_BACKEND,
    DEFAULT_FISTA_PROX_ORACLE,
    DEFAULT_FISTA_RESTART,
    DEFAULT_IMPLEMENTATION,
    DEFAULT_PAVA,
    PerspectiveRelaxation,
    available_backends,
    registered_backends,
    solve_relaxation,
)
from .certificate import SafeDualCertificate
from .problem import (
    MarkowitzInstance,
    evaluate_solution,
    load_instance_bundle,
    perspective_value,
    save_instance_bundle,
)
from .result import RelaxationResult
from .native import (
    AUTO_IMPLEMENTATION,
    IMPLEMENTATIONS,
    native_availability,
    native_available,
)
from .state import RelaxationState
from .fista import FISTA_PROX_ORACLES, FISTA_RESTART_STRATEGIES

__all__ = [
    "BACKENDS",
    "AUTO_IMPLEMENTATION",
    "DEFAULT_BACKEND",
    "DEFAULT_FISTA_PROX_ORACLE",
    "DEFAULT_FISTA_RESTART",
    "DEFAULT_IMPLEMENTATION",
    "DEFAULT_PAVA",
    "FISTA_PROX_ORACLES",
    "FISTA_RESTART_STRATEGIES",
    "MarkowitzInstance",
    "IMPLEMENTATIONS",
    "PerspectiveRelaxation",
    "RelaxationResult",
    "RelaxationState",
    "SafeDualCertificate",
    "available_backends",
    "evaluate_solution",
    "load_instance_bundle",
    "native_availability",
    "native_available",
    "perspective_value",
    "registered_backends",
    "save_instance_bundle",
    "solve_relaxation",
]
