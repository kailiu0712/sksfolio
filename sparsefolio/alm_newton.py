"""Augmented-Lagrangian Newton solver using only the scalar perspective prox.

Linear exposures enter a smooth squared-distance penalty.  Inner generalized
Newton equations are solved by matrix-free GMRES, with a forward-backward
envelope line search and proximal-gradient fallback.  No affine-constrained
proximal subproblem and no external optimization solver are used here.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import LinearOperator, gmres

try:
    from .core import prox, evaluate
except ImportError:
    from .core import prox, evaluate


class _Timeout(Exception):
    pass


def _norm_squared(a):
    if not a.size:
        return 0.0
    gram = a @ a.T if a.shape[0] <= a.shape[1] else a.T @ a
    return max(0.0, float(eigh(gram, subset_by_index=[len(gram)-1]*2,
                                eigvals_only=True, check_finite=False)[0])) * (1+1e-12)


def _apply_j(d, h, denominator, v):
    out = d*v
    if denominator > 0:
        out -= h * (h @ v / denominator)
    return out


def solve(problem, mode='alm_newton', time_limit=15, tol=1e-6,
          feas_tol=1e-7, max_iter=1000):
    """Return a root-relaxation point with independently assessed accuracy.

    max_iter counts attempted inner Newton steps.  The penalty grows only when
    actual exposure infeasibility exceeds the requested feasibility tolerance.
    Time checks occur between operations and in each GMRES iteration.
    """
    if mode != 'alm_newton':
        raise ValueError('mode must be alm_newton')
    start = time.perf_counter()
    deadline = start + float(time_limit)
    p = problem
    B, K = p.B, p.K
    n, m = p.n, len(p.lo)
    lf, k2 = _norm_squared(B), _norm_squared(K)
    rho = max(1e-3, lf/max(k2, 1e-12))
    initial_rho = rho
    x = np.minimum(p.u, np.full(n, 1/n))
    y = np.zeros(m)
    equality = p.lo == p.hi
    raw_K = p.raw_K if p.raw_K is not None else p.K
    raw_lo = p.raw_lo if p.raw_lo is not None else p.lo
    raw_hi = p.raw_hi if p.raw_hi is not None else p.hi
    counts = dict(mode=mode, prox_calls=0, krylov_iterations=0,
                  krylov_failures=0, newton_steps=0, gradient_fallbacks=0,
                  backtracks=0, penalty_increases=0, outer_iterations=0,
                  matrix_vector_products=0, max_active_constraints=0,
                  linear_seconds=0.0, setup_seconds=time.perf_counter()-start,
                  rho_initial=rho, smooth_lipschitz=lf, constraint_norm_squared=k2)
    history = []
    best, best_merit = None, np.inf
    status = 'iteration_limit'
    previous_feasibility = np.inf
    total, outer = 0, 0
    inner_initial_tolerance = 0.01 * max(1.0, np.linalg.norm(p.c, np.inf))

    def check_time():
        if time.perf_counter() >= deadline:
            raise _Timeout

    def certificate(xx, yy, mapping):
        nonlocal best, best_merit
        metrics = evaluate(p, xx, yy)
        merit = max(metrics['feasibility']/feas_tol, metrics['rel_gap']/tol)
        history.append(dict(iteration=total, outer_iteration=outer,
                            seconds=time.perf_counter()-start, rho=rho,
                            mapping=float(mapping), **metrics))
        if best is None or merit < best_merit:
            best = (xx.copy(), yy.copy(), metrics.copy())
            best_merit = merit
        return merit <= 1.0

    def point(xx, multipliers, penalty, tau):
        check_time()
        bx = B @ xx
        shifted = K @ xx + multipliers / penalty
        clipped = np.clip(shifted, p.lo, p.hi)
        distance = shifted-clipped
        normal = penalty*distance
        gradient = B.T @ bx + p.c + K.T @ normal
        px, z, theta, d, h, denominator = prox(xx-tau*gradient, tau, p.gamma, p.u, p.k)
        counts['prox_calls'] += 1
        residual = xx-px
        gv = 0.5*np.sum(p.gamma*np.divide(px*px, z, out=np.zeros_like(px), where=z>0))
        smooth = 0.5*(bx@bx) + p.c@xx + 0.5*penalty*(distance@distance)
        envelope = smooth + gv - gradient@residual + (residual@residual)/(2*tau)
        active = equality | (shifted < p.lo) | (shifted > p.hi)
        counts['max_active_constraints'] = max(counts['max_active_constraints'], int(active.sum()))
        return dict(px=px, residual=residual, envelope=float(envelope), d=d,
                    h=h, denominator=denominator, active=active)

    def candidate_y(xx, multipliers, penalty):
        shifted = K @ xx + multipliers/penalty
        return penalty*(shifted-np.clip(shifted,p.lo,p.hi))

    certificate(x,y,np.inf)
    try:
        while total < max_iter:
            check_time()
            outer += 1
            tau = 0.95/max(lf+rho*k2,1e-14)
            # Avoid asking the scalar prox for a residual below its numerical
            # resolution when a large penalty makes tau small.  The external
            # certificate still independently enforces the requested accuracy.
            inner_tolerance = max(1e-9, 2e-13/tau,
                                  inner_initial_tolerance * 0.4**min(outer,60))
            current = point(x,y,rho,tau)
            inner_limit = min(80, max_iter-total)
            for _ in range(inner_limit):
                mapping = np.linalg.norm(current['residual'],np.inf)/tau
                yc = candidate_y(current['px'],y,rho)
                if certificate(current['px'],yc,mapping):
                    status='converged'
                    break
                if mapping <= inner_tolerance:
                    break
                check_time()
                ka = K[current['active']]

                def hv(v):
                    counts['matrix_vector_products'] += 1
                    return B.T@(B@v) + rho*(ka.T@(ka@v))

                d,h,denominator = current['d'],current['h'],current['denominator']
                residual = current['residual']
                # Perspective pieces can be flat in many directions.  A
                # residual-proportional regularizer is too small on those
                # pieces and can create enormous nullspace Newton steps.
                regularization = min(1e-2,max(1e-10,0.1*np.sqrt(np.linalg.norm(residual))))

                def action(v):
                    check_time()
                    return (1+regularization)*v-_apply_j(d,h,denominator,v-tau*hv(v))

                diagonal_hessian = np.sum(B*B,axis=0)+rho*np.sum(ka*ka,axis=0)
                diagonal = np.maximum(1-d+tau*d*diagonal_hessian+regularization,1e-8)
                operator = LinearOperator((n,n),matvec=action,dtype=float)
                preconditioner = LinearOperator((n,n),matvec=lambda v:v/diagonal,dtype=float)

                def callback(_):
                    counts['krylov_iterations'] += 1
                    check_time()

                linear_start=time.perf_counter()
                direction, info = gmres(operator,-residual,M=preconditioner,
                                        rtol=min(0.1,max(1e-9,np.linalg.norm(residual)**0.5)),
                                        atol=1e-14,restart=min(50,n),maxiter=10,
                                        callback=callback,callback_type='pr_norm')
                counts['linear_seconds'] += time.perf_counter()-linear_start
                counts['krylov_failures'] += int(info != 0)
                counts['newton_steps'] += 1
                total += 1
                radius=max(0.1,2*np.linalg.norm(x))
                if np.linalg.norm(direction)>radius:
                    direction*=radius/np.linalg.norm(direction)
                envelope_gradient=(residual-tau*hv(residual))/tau
                directional=float(envelope_gradient@direction)
                if not np.all(np.isfinite(direction)):
                    direction=-residual
                    directional=float(envelope_gradient@direction)
                    counts['gradient_fallbacks'] += 1
                accepted=False
                alpha=1.0
                for _ in range(22):
                    trial_x=x+alpha*direction
                    trial=point(trial_x,y,rho,tau)
                    allowance=5e-16*max(1.0,abs(current['envelope']))
                    residual_decrease = np.linalg.norm(trial['residual']) <= (1-1e-3*alpha)*np.linalg.norm(residual)
                    envelope_decrease = directional < 0 and trial['envelope']<=current['envelope']+1e-4*alpha*directional+allowance
                    if residual_decrease or envelope_decrease:
                        x,current=trial_x,trial
                        accepted=True
                        break
                    alpha*=0.5
                    counts['backtracks'] += 1
                if not accepted:
                    x=current['px'].copy()
                    current=point(x,y,rho,tau)
                    counts['gradient_fallbacks'] += 1
            x=current['px'].copy()
            y=candidate_y(x,y,rho)
            counts['outer_iterations']=outer
            if status=='converged':
                break
            if certificate(x,y,np.linalg.norm(current['residual'],np.inf)/tau):
                status='converged'
                break
            ax=raw_K@x
            feasibility=max(0.0,float(np.max(np.maximum(raw_lo-ax,ax-raw_hi))))
            if feasibility>feas_tol and feasibility>0.65*previous_feasibility:
                rho=min(3*rho,initial_rho*1e8)
                counts['penalty_increases'] += 1
            previous_feasibility=feasibility
    except _Timeout:
        status='time_limit'
    if best_merit<=1:
        status='converged'
    bx,by,metrics=best
    counts.update(seconds=time.perf_counter()-start,rho_final=rho,
                  certificates=len(history),selected_merit=best_merit)
    return dict(x=bx,y=by,iterations=total,status=status,diagnostics=counts,
                history=history,metrics=metrics,seconds=counts['seconds'])
