"""Small portable collection of sparse portfolio root-relaxation solvers."""

__version__ = "0.1.0"


def make_problem(**kwargs):
    from .datasets import make_problem as build
    return build(**kwargs)


def solve(problem, method="alm_polished", **kwargs):
    from .api import solve as run
    return run(problem, method=method, **kwargs)


def warmup():
    from .api import warmup as compile_kernels
    return compile_kernels()


def methods(include_mosek=False):
    from .api import CUSTOM_METHODS, MOSEK_METHODS
    return CUSTOM_METHODS + (MOSEK_METHODS if include_mosek else ())


def __getattr__(name):
    if name == "Problem":
        from .core import Problem
        return Problem
    raise AttributeError(name)
