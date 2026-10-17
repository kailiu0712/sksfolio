"""Independent continuous conic reference for the perspective root relaxation."""
from __future__ import annotations

from time import perf_counter
import numpy as np

from .core import evaluate


def solve_mosek(problem, time_limit=30, tol=1e-9):
    """Solve with continuous selection variables and one MOSEK CPU thread.

    The rotated cones impose 2*t_i*z_i >= gamma_i*x_i^2 and
    2*r >= ||B*x||^2.  The objective is r + sum(t) + c.T*x.
    Returned y uses positive upper-bound normals and negative lower-bound
    normals, matching core.evaluate.  Total time includes model construction.
    """
    start = perf_counter()
    from mosek.fusion import (AccSolutionStatus, Domain, Expr, Model,
                              ObjectiveSense, SolutionStatus)
    p = problem
    with Model("independent_perspective_root") as model:
        x = model.variable("x", p.n, Domain.greaterThan(0.0))
        z = model.variable("z", p.n, Domain.inRange(0.0, 1.0))
        t = model.variable("t", p.n, Domain.greaterThan(0.0))
        risk = model.variable("risk", 1, Domain.greaterThan(0.0))
        model.constraint("cardinality", Expr.sum(z), Domain.lessThan(float(p.k)))
        model.constraint("perspective_caps", Expr.sub(x, Expr.mulElm(p.u, z)),
                         Domain.lessThan(0.0))
        model.constraint("perspective", Expr.hstack(t, z, Expr.mulElm(np.sqrt(p.gamma), x)),
                         Domain.inRotatedQCone())
        model.constraint("factor_risk", Expr.vstack(risk, Expr.constTerm(1.0), Expr.mul(p.B, x)),
                         Domain.inRotatedQCone())
        model.objective(ObjectiveSense.Minimize,
                        Expr.add(Expr.add(risk, Expr.sum(t)), Expr.dot(p.c, x)))

        Kx = Expr.mul(p.K, x)
        equality = np.flatnonzero(p.lo == p.hi)
        lower = np.flatnonzero(np.isfinite(p.lo) & (p.lo != p.hi))
        upper = np.flatnonzero(np.isfinite(p.hi) & (p.lo != p.hi))
        constraints = []
        if len(equality):
            constraints.append((equality, model.constraint("equalities", Kx.pick(equality.tolist()),
                                                        Domain.equalsTo(p.lo[equality]))))
        if len(lower):
            constraints.append((lower, model.constraint("lower_bounds", Kx.pick(lower.tolist()),
                                                     Domain.greaterThan(p.lo[lower]))))
        if len(upper):
            constraints.append((upper, model.constraint("upper_bounds", Kx.pick(upper.tolist()),
                                                     Domain.lessThan(p.hi[upper]))))

        model.setSolverParam("numThreads", 1)
        model.setSolverParam("optimizerMaxTime", float(time_limit))
        for parameter in ("intpntCoTolPfeas", "intpntCoTolDfeas", "intpntCoTolRelGap", "intpntCoTolMuRed"):
            model.setSolverParam(parameter, float(tol))
        build_seconds = perf_counter() - start
        solve_start = perf_counter()
        model.solve()
        solve_wall_seconds = perf_counter() - solve_start
        model.acceptedSolutionStatus(AccSolutionStatus.Anything)
        primal_status = model.getPrimalSolutionStatus()
        dual_status = model.getDualSolutionStatus()
        solver_seconds = float(model.getSolverDoubleInfo("optimizerTime"))
        iterations = int(model.getSolverIntInfo("intpntIter"))
        xv = np.asarray(x.level(), dtype=float)
        yv = np.zeros(len(p.lo))
        for indices, constraint in constraints:
            yv[indices] -= np.asarray(constraint.dual(), dtype=float)
        cone_objective = float(model.primalObjValue())
        cone_bound = float(model.dualObjValue())
        diagnostic = evaluate(p, xv, yv)
        diagnostic.update(conic_objective=cone_objective, conic_lower_bound=cone_bound,
                          conic_gap=cone_objective-cone_bound, build_seconds=build_seconds,
                          solve_wall_seconds=solve_wall_seconds,
                          primal_status=str(primal_status), dual_status=str(dual_status),
                          problem_status=str(model.getProblemStatus()), threads=1)
        status = "optimal" if primal_status == SolutionStatus.Optimal else str(primal_status)
        return dict(x=xv, y=yv, objective=diagnostic["objective"],
                    lower_bound=diagnostic["lower_bound"], solver_seconds=solver_seconds,
                    total_seconds=perf_counter()-start, status=status, iterations=iterations,
                    diagnostics=diagnostic)
