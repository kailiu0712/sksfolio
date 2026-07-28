"""MOSEK backends."""

from .julia import solve as solve_julia
from .python import solve as solve_python

__all__ = ["solve_julia", "solve_python"]
