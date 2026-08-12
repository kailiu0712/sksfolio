"""Building blocks for accelerated proximal-gradient portfolio solvers."""

from .budget_prox import (
    BudgetProxResult,
    prox_budget,
    prox_budget_details,
)
from .linear_prox import (
    LinearConstraintProx,
    LinearProxResult,
    prox_linear,
    prox_linear_details,
)
from .majorization_qp import (
    MajorizationQPOracle,
    MajorizationQPResult,
    MajorizationQPWarmStart,
)
from .solver import solve_fista


FISTA_PROX_ORACLES = (
    "auto",
    "pava",
    "budget",
    "dual_fista",
    "dual_lbfgs",
    "majorization_qp",
)
FISTA_RESTART_STRATEGIES = (
    "none",
    "gradient",
    "function",
    "periodic",
    "hinder_lubin",
    "primal_dual_gap",
)

__all__ = [
    "BudgetProxResult",
    "FISTA_PROX_ORACLES",
    "FISTA_RESTART_STRATEGIES",
    "LinearConstraintProx",
    "LinearProxResult",
    "MajorizationQPOracle",
    "MajorizationQPResult",
    "MajorizationQPWarmStart",
    "prox_budget",
    "prox_budget_details",
    "prox_linear",
    "prox_linear_details",
    "solve_fista",
]
