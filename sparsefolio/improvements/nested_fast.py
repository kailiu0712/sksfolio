"""Structured linear algebra for the nested constrained-prox Newton solver.

Both variants use the same dual active-set Newton method, with a split-dual
L-BFGS-B rescue after ten unsuccessful steps, for the constrained proximal map.
``nested_fast_prox`` changes only the scalar prox of the original Krylov method.
``nested_fast_outer`` adds exact structured outer solves, and ``nested_fast`` also
uses support-aware inner dual solves.  ``nested_fast_inexact`` uses both improvements
and the original experimental inexact forcing schedule.
A forward-backward-envelope line search globalizes the outer
iteration.  These are experimental implementations, not reproductions of a
particular paper or imports from the downloaded portfolio package.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
from scipy import linalg
from scipy.optimize import minimize
from scipy.sparse.linalg import LinearOperator, gmres

try:
    from .fast_core import prox
    from .inner_dual_fast import solve_inner
except ImportError:
    from fast_core import prox
    from inner_dual_fast import solve_inner
from ..core import evaluate


class _Timeout(Exception):
    pass


def _apply_j(d, h, denom, v):
    """Apply the diagonal-minus-rank-one generalized proximal Jacobian."""
    if v.ndim == 1:
        out = d * v
        return out - h * (h @ v / denom) if denom > 0 else out
    out = d[:, None] * v
    return out - h[:, None] * ((h @ v)[None, :] / denom) if denom > 0 else out


def _g_value(x, z, gamma):
    return 0.5 * float(np.sum(gamma * np.divide(x * x, z, out=np.zeros_like(x), where=z > 0)))


def _support(y, lo, hi):
    pos, neg = y > 0, y < 0
    return float(y[pos] @ hi[pos] + y[neg] @ lo[neg])


def _support_direction(y, dy, lo, hi):
    """Directional derivative without evaluating zero times infinity."""
    up = (y > 0) | ((y == 0) & (dy > 0))
    low = (y < 0) | ((y == 0) & (dy < 0))
    return float(dy[up] @ hi[up] + dy[low] @ lo[low])


def _domain(y, lo, hi):
    out = y.copy()
    out[np.isneginf(lo)] = np.maximum(out[np.isneginf(lo)], 0.0)
    out[np.isposinf(hi)] = np.minimum(out[np.isposinf(hi)], 0.0)
    return out


def _spectral_norm_squared(mat, steps=30):
    if not mat.size:
        return 0.0
    # An all-ones power start can be orthogonal to centered factor rows.
    # The small row Gram matrix gives an actual spectral bound for these cases.
    gram = mat @ mat.T
    return max(0.0, float(linalg.eigvalsh(gram, subset_by_index=[len(gram)-1, len(gram)-1],
                                       check_finite=False)[0]))


def _psd_solve(matrix, rhs, regularization):
    if not len(rhs):
        return rhs.copy()
    matrix = 0.5 * (matrix + matrix.T)
    matrix.flat[::len(rhs) + 1] += regularization
    try:
        return linalg.cho_solve(linalg.cho_factor(matrix, lower=True, check_finite=False), rhs, check_finite=False)
    except linalg.LinAlgError:
        return linalg.lstsq(matrix, rhs, cond=1e-12, check_finite=False)[0]


class _ConstrainedProx:
    def __init__(self, problem, tau, deadline, counters, fast_inner=False):
        self.p = problem
        self.tau = tau
        self.deadline = deadline
        self.c = counters
        self.fast_inner = fast_inner
        self.K = np.asarray(problem.K)
        self.lo, self.hi = np.asarray(problem.lo), np.asarray(problem.hi)
        self.eq = np.isfinite(self.lo) & (self.lo == self.hi)
        self.lipschitz = max(tau * _spectral_norm_squared(self.K) * 1.05, 1e-12)

    def base(self, v, y):
        if time.perf_counter() >= self.deadline:
            raise _Timeout
        base_start=time.perf_counter()
        self.c['prox_calls'] += 1
        shifted=v-self.tau*(self.K.T@y)
        prox_start=time.perf_counter()
        x, z, theta, d, h, denom = prox(shifted, self.tau,
                                        self.p.gamma, self.p.u, self.p.k)
        self.c['scalar_prox_seconds']+=time.perf_counter()-prox_start
        kx = self.K @ x
        residual = kx - np.clip(kx + y, self.lo, self.hi)
        gv = _g_value(x, z, self.p.gamma)
        # This is the exact concave dual objective of the proximal problem.
        dual = gv + float((x-v) @ (x-v)) / (2*self.tau) + y @ kx - _support(y, self.lo, self.hi)
        self.c['base_evaluation_seconds']+=time.perf_counter()-base_start
        return dict(x=x, z=z, theta=theta, d=d, h=h, denom=denom,
                    kx=kx, residual=residual, dual=float(dual), g=gv)

    def __call__(self, v, warm, tolerance=1e-10, max_iterations=250):
        y = _domain(np.asarray(warm), self.lo, self.hi)
        current = self.base(v, y)
        m = self.K.shape[0]
        if m == 0:
            current.update(y=y, inner_residual=0.0, inner_status='converged', active=np.zeros(0, dtype=bool))
            return current
        status = 'iteration_limit'
        for iteration in range(max_iterations):
            if iteration == 10:
                y, current = self.rescue(v, y, max_iterations=500, tolerance=tolerance)
            self.c['inner_iterations'] += 1
            residual = current['residual']
            norm = float(np.linalg.norm(residual, np.inf))
            if norm <= tolerance:
                status = 'converged'
                break
            active = self.eq | (current['kx'] + y > self.hi) | (current['kx'] + y < self.lo)
            self.c['max_active_constraints'] = max(self.c['max_active_constraints'], int(active.sum()))
            self.c['active_constraint_sum'] += int(active.sum())
            dy = -y.copy()
            if np.any(active):
                linear_start = time.perf_counter()
                ka = self.K[active]
                inactive = dy.copy()
                inactive[active] = 0.0
                correction = self.tau * ka @ _apply_j(current['d'], current['h'], current['denom'], self.K.T @ inactive)
                rhs = residual[active] - correction
                if self.fast_inner:
                    positive = current['d'] > 0
                    kap = ka[:,positive]
                    diagonal = self.tau*((kap*kap)@current['d'][positive])
                    if current['denom'] > 0:
                        diagonal -= self.tau*(kap@current['h'][positive])**2/current['denom']
                else:
                    jka = _apply_j(current['d'], current['h'], current['denom'], ka.T)
                    matrix = self.tau * (ka @ jka)
                    diagonal = np.diag(matrix)
                diagscale = max(float(np.max(diagonal)), self.lipschitz*1e-4, 1e-12)
                reg = diagscale * min(1e-3, max(1e-12, norm*1e-2))
                if self.fast_inner:
                    dy[active], detail = solve_inner(ka,current['d'],current['h'],current['denom'],self.tau,reg,rhs)
                    self.c['inner_structured_solves'] += 1
                    self.c['inner_support_qr_solves'] += int(detail.get('method') == 'support_qr')
                    self.c['inner_active_cholesky_solves'] += int(detail.get('method') == 'active_cholesky')
                    self.c['max_inner_system']=max(self.c['max_inner_system'],detail.get('system_dimension',len(ka)))
                    self.c['inner_support_sum']+=int(positive.sum())
                    self.c['max_inner_support']=max(self.c['max_inner_support'],int(positive.sum()))
                else:
                    dy[active] = _psd_solve(matrix, rhs, reg)
                self.c['inner_linear_seconds'] += time.perf_counter()-linear_start
                self.c['inner_factorizations'] += 1
            # Flat proximal pieces can produce enormous regularized Newton
            # directions.  A finite trust radius lets the line search leave
            # such a piece without spending dozens of iterations halving it.
            radius = 10.0 * (1.0 + np.linalg.norm(y, np.inf) + np.linalg.norm(v, np.inf)/self.tau)
            direction_norm = np.linalg.norm(dy, np.inf)
            if direction_norm > radius:
                dy *= radius/direction_norm
            # Orthant/domain projection is necessary for one-sided intervals.
            dy = _domain(y + dy, self.lo, self.hi) - y
            derivative = float(current['kx'] @ dy - _support_direction(y, dy, self.lo, self.hi))
            accepted = False
            if np.isfinite(derivative) and derivative > 0:
                alpha = 1.0
                for _ in range(24):
                    yt = _domain(y + alpha * dy, self.lo, self.hi)
                    trial = self.base(v, yt)
                    # The residual alternative prevents roundoff in a nearly
                    # flat dual objective from rejecting a successful Newton step.
                    roundoff = 1e-14 * max(1.0, abs(current['dual']))
                    improved_dual = trial['dual'] >= current['dual'] + 1e-4*alpha*derivative - roundoff
                    improved_res = np.linalg.norm(trial['residual'], np.inf) <= (1-1e-3*alpha)*norm
                    if improved_dual and (improved_res or trial['dual'] > current['dual'] + roundoff):
                        y, current, accepted = yt, trial, True
                        break
                    alpha *= 0.5
                    self.c['inner_backtracks'] += 1
            if not accepted:
                # Proximal gradient on the nonsmooth dual.  The support-function
                # prox is computed using Moreau's identity and a box projection.
                self.c['inner_gradient_fallbacks'] += 1
                step = 1.0/self.lipschitz
                for _ in range(30):
                    w = y + step * current['kx']
                    yt = _domain(w - step * np.clip(w/step, self.lo, self.hi), self.lo, self.hi)
                    trial = self.base(v, yt)
                    if trial['dual'] >= current['dual'] - 1e-14*max(1.0, abs(current['dual'])):
                        y, current = yt, trial
                        break
                    step *= 0.5
                else:
                    status = 'stalled'
                    break
        active = self.eq | (current['kx'] + y > self.hi) | (current['kx'] + y < self.lo)
        current.update(y=y, inner_residual=float(np.linalg.norm(current['residual'], np.inf)),
                       inner_status=status, active=active)
        if current['inner_residual'] <= tolerance:
            current['inner_status'] = 'converged'
        return current

    def rescue(self, v, warm, max_iterations=500, tolerance=1e-10):
        """Smooth split-dual L-BFGS-B rescue for difficult active-set changes.

        Each finite upper and lower bound has its own nonnegative multiplier;
        equalities have free multipliers.  This is the same proximal subproblem.
        Newton then polishes the result to the requested residual accuracy.
        """
        eq = np.flatnonzero(self.eq)
        up = np.flatnonzero(np.isfinite(self.hi) & ~self.eq)
        low = np.flatnonzero(np.isfinite(self.lo) & ~self.eq)
        ne, nu = len(eq), len(up)
        initial = np.concatenate((warm[eq], np.maximum(warm[up], 0), np.maximum(-warm[low], 0)))
        bounds = [(None, None)]*ne + [(0, None)]*(nu+len(low))
        support_coefficients = np.concatenate((self.hi[eq], self.hi[up], -self.lo[low]))
        self.c['inner_lbfgs_rescues'] += 1
        def unpack(variables):
            yy = np.zeros_like(warm)
            yy[eq] = variables[:ne]
            yy[up] += variables[ne:ne+nu]
            yy[low] -= variables[ne+nu:]
            return yy
        def objective(variables):
            yy = unpack(variables)
            result = self.base(v, yy)
            value = -result['dual'] + variables @ support_coefficients - _support(yy, self.lo, self.hi)
            kx = result['kx']
            gradient = np.concatenate((self.hi[eq]-kx[eq], self.hi[up]-kx[up], kx[low]-self.lo[low]))
            return value, gradient
        def callback(_):
            self.c['inner_lbfgs_iterations'] += 1
        rescue_start=time.perf_counter()
        try:
            result = minimize(objective, initial, method='L-BFGS-B', jac=True, bounds=bounds,
                              callback=callback, options=dict(maxiter=max_iterations, maxls=50,
                                                              ftol=1e-15, gtol=tolerance*0.1, maxcor=20))
        finally:
            self.c['inner_lbfgs_seconds']+=time.perf_counter()-rescue_start
        yy = unpack(result.x)
        return yy, self.base(v, yy)

    def jacobian(self, result):
        """Return the action of the constrained proximal Jacobian.

        A spectral pseudoinverse handles redundant binding exposure rows.
        Its largest stored objects are n-by-active_rows and active_rows squared.
        """
        d, h, denom = result['d'], result['h'], result['denom']
        ka = self.K[result['active']]
        if len(ka) == 0:
            return lambda v: _apply_j(d, h, denom, v)
        jka = _apply_j(d, h, denom, ka.T)
        gram = ka @ jka
        values, vectors = linalg.eigh(0.5*(gram+gram.T), check_finite=False)
        threshold = max(float(values[-1])*1e-11, 1e-14)
        keep = values > threshold
        basis = vectors[:, keep]
        inverse = 1.0/values[keep]
        self.c['derivative_factorizations'] += 1
        self.c['max_derivative_rank'] = max(self.c['max_derivative_rank'], int(keep.sum()))
        def apply(v):
            jv = _apply_j(d, h, denom, v)
            coeff = basis.T @ (ka @ jv)
            coeff = inverse * coeff if v.ndim == 1 else inverse[:, None] * coeff
            return jv - jka @ (basis @ coeff)
        return apply

    def jacobian_factors(self, result):
        """Return support S and J_Q[S,S]=diag(d[S])-W W.T.

        A thin singular decomposition is used when there are fewer primal
        derivative coordinates than binding rows.  Otherwise the small row
        Gram eigendecomposition is used.  Both use the original cutoff.
        """
        d,h,den = result['d'],result['h'],result['denom']
        support = np.flatnonzero(d > 0)
        ds,hs = d[support],h[support]
        ka = self.K[result['active']][:,support]
        pieces=[]
        if den > 0:
            pieces.append(hs[:,None]/np.sqrt(den))
        if len(ka) and len(support):
            if len(support) < len(ka):
                sd=np.sqrt(ds)
                v=hs/sd
                alpha=1/(den*(1+np.sqrt(max(0,1-v@v/den)))) if den>0 else 0.0
                af=ka*sd
                if den>0:
                    af-=alpha*(ka@hs)[:,None]*v[None,:]
                _,singular,vt=linalg.svd(af,full_matrices=False,check_finite=False)
                cutoff=np.sqrt(max(singular[0]**2*1e-11,1e-14)) if len(singular) else np.inf
                basis=vt[singular>cutoff].T
                factor=sd[:,None]*basis
                if den>0:
                    factor-=alpha*hs[:,None]*(v@basis)[None,:]
                self.c['derivative_support_svd'] += 1
            else:
                jka=ds[:,None]*ka.T
                if den>0:
                    jka-=hs[:,None]*(hs@ka.T)[None,:]/den
                gram=ka@jka
                values,vectors=linalg.eigh(.5*(gram+gram.T),check_finite=False)
                keep=values>max(values[-1]*1e-11,1e-14)
                factor=jka@(vectors[:,keep]/np.sqrt(values[keep])[None,:])
            pieces.append(factor)
            self.c['derivative_factorizations'] += 1
            self.c['max_derivative_rank']=max(self.c['max_derivative_rank'],factor.shape[1])
        w=np.concatenate(pieces,axis=1) if pieces else np.empty((len(support),0))
        return support,ds,w


def _outer_direction(B,support,d,W,tau,regularization,rhs):
    """Exact solve of [(1+r)I-J_Q+tau*J_Q*B.T*B] direction=rhs."""
    a=1+regularization
    answer=rhs/a
    if not len(support):
        return answer,0,'empty'
    bs=np.ascontiguousarray(B[:,support])
    if len(support)<=W.shape[1]+len(B):
        jq=np.diag(d)-W@W.T
        fixed=answer.copy();fixed[support]=0
        target=rhs[support]-tau*jq@(bs.T@(B@fixed))
        matrix=a*np.eye(len(support))-jq+tau*jq@(bs.T@bs)
        answer[support]=linalg.solve(matrix,target,check_finite=False)
        return answer,len(support),'support'
    inv=1/(a-d)
    iw=inv[:,None]*W
    small=np.eye(W.shape[1])+W.T@iw
    factor=linalg.cho_factor(.5*(small+small.T),lower=True,check_finite=False)
    def cinv(v):
        iv=inv*v if v.ndim==1 else inv[:,None]*v
        return iv-iw@linalg.cho_solve(factor,W.T@iv,check_finite=False)
    answer[support]=cinv(rhs[support])
    tbt=tau*(a*cinv(bs.T)-bs.T)
    schur=np.eye(len(B))+bs@tbt
    coef=linalg.solve(schur,B@answer,check_finite=False)
    answer[support]-=tbt@coef
    return answer,max(W.shape[1],len(B)),'woodbury'


def solve(problem, mode='nested_fast', time_limit=20, tol=1e-6, feas_tol=1e-7, max_iter=200):
    """Solve and certify the root relaxation, returning the feasible prox point.

    ``y`` multiplies the supplied K in the original objective's units.  Stopping
    uses the independently computed primal-dual certificate, never a reference
    objective.  The time limit is checked between numerical operations.
    """
    if mode not in ('nested_fast_prox','nested_fast_outer','nested_fast','nested_fast_inexact'):
        raise ValueError('unknown nested-fast mode')
    inexact = mode == 'nested_fast_inexact'
    direct_outer=mode!='nested_fast_prox'
    started = time.perf_counter()
    deadline = started + time_limit
    b, c = np.asarray(problem.B), np.asarray(problem.c)
    n, m = int(problem.n), len(problem.lo)
    counters = dict(prox_calls=0, inner_iterations=0, inner_factorizations=0,
                    inner_backtracks=0, inner_gradient_fallbacks=0,
                    inner_lbfgs_rescues=0, inner_lbfgs_iterations=0,
                    max_active_constraints=0, active_constraint_sum=0,
                    derivative_factorizations=0, max_derivative_rank=0,
                    outer_factorizations=0, outer_backtracks=0,
                    outer_gradient_fallbacks=0, krylov_iterations=0,
                    krylov_failures=0, matrix_vector_products=0,
                    inner_prox_seconds=0.0, outer_linear_seconds=0.0,
                    inner_linear_seconds=0.0,inner_structured_solves=0,
                    inner_support_qr_solves=0,outer_support_solves=0,
                    inner_active_cholesky_solves=0,max_inner_system=0,
                    inner_support_sum=0,max_inner_support=0,
                    scalar_prox_seconds=0.0,base_evaluation_seconds=0.0,
                    inner_lbfgs_seconds=0.0,
                    outer_woodbury_solves=0,max_outer_system=0,
                    derivative_support_svd=0)
    lipschitz = _spectral_norm_squared(b) * 1.05
    tau = 0.9 / max(lipschitz, 1e-8)
    cp = _ConstrainedProx(problem, tau, deadline, counters,fast_inner=mode in ('nested_fast','nested_fast_inexact'))
    x = np.full(n, 1.0/n)
    y = np.zeros(m)
    history = []
    last = None
    status = 'iteration_limit'
    iteration = 0
    hessian = None
    def hv(v):
        counters['matrix_vector_products'] += 1
        return hessian @ v if hessian is not None else b.T @ (b @ v)
    strict_inner_tolerance = min(1e-10, feas_tol*0.05)
    def point(xx, yy, max_inner=250, inner_tolerance=None):
        grad = b.T @ (b @ xx) + c
        if inner_tolerance is None:
            inner_tolerance = strict_inner_tolerance
        inner_started = time.perf_counter()
        try:
            out = cp(xx-tau*grad, yy, tolerance=inner_tolerance, max_iterations=max_inner)
        finally:
            counters['inner_prox_seconds'] += time.perf_counter()-inner_started
        residual = xx-out['x']
        envelope = 0.5*float((b@xx) @ (b@xx)) + c@xx + out['g'] - grad@residual + residual@residual/(2*tau)
        out.update(residual_outer=residual, envelope=float(envelope), outer_x=xx,
                   inner_tolerance=inner_tolerance)
        return out
    try:
        current = point(x, y, inner_tolerance=1e-4 if inexact else strict_inner_tolerance)
        for iteration in range(max_iter+1):
            # Experimental inexact forcing schedule.  Refinement uses the
            # outer residual squared and the same final shared certificate.
            # No convergence theorem for this schedule is asserted here.
            if inexact:
                requested = max(strict_inner_tolerance,
                                min(1e-4, 0.1*np.linalg.norm(current['residual_outer'], np.inf)**2))
                if current['inner_tolerance'] > 2*requested:
                    current = point(x, current['y'], inner_tolerance=requested)
            last = current
            certificate = evaluate(problem, current['x'], current['y'])
            history.append(dict(iteration=iteration, seconds=time.perf_counter()-started,
                                residual=float(np.linalg.norm(current['residual_outer'], np.inf)),
                                inner_residual=current['inner_residual'],
                                inner_tolerance=current['inner_tolerance'],
                                **{key: float(certificate[key]) for key in ('objective', 'feasibility', 'dual_gap', 'rel_gap')}))
            if certificate['rel_gap'] <= tol and certificate['feasibility'] <= feas_tol:
                status = 'converged'
                break
            if time.perf_counter() >= deadline:
                status = 'time_limit'
                break
            if iteration == max_iter:
                break
            if current['inner_residual'] > max(1e-7, feas_tol, current['inner_tolerance']*1.1):
                status = 'inner_iteration_limit'
                break
            linear_started = time.perf_counter()
            if direct_outer:
                support,ds,w=cp.jacobian_factors(current)
            else:
                jq = cp.jacobian(current)
            r = current['residual_outer']
            grad_envelope = (r-tau*hv(r))/tau
            regularization = min(1e-4, max(1e-12, 1e-2*np.linalg.norm(r, np.inf)))
            if direct_outer:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', linalg.LinAlgWarning)
                    try:
                        direction,dimension,branch=_outer_direction(b,support,ds,w,tau,regularization,-r)
                        counters['max_outer_system']=max(counters['max_outer_system'],dimension)
                        counters['outer_support_solves']+=int(branch=='support')
                        counters['outer_woodbury_solves']+=int(branch=='woodbury')
                    except linalg.LinAlgError:
                        direction = -r
                counters['outer_factorizations'] += 1
            else:
                def action(v):
                    return (1+regularization)*v - jq(v-tau*hv(v))
                operator = LinearOperator((n,n), matvec=action, dtype=float)
                def callback(_):
                    counters['krylov_iterations'] += 1
                    if time.perf_counter() >= deadline:
                        raise _Timeout
                direction, info = gmres(operator, -r, rtol=min(0.1, max(1e-9, np.linalg.norm(r)**0.5)),
                                        atol=1e-13, restart=min(50,n), maxiter=10,
                                        callback=callback, callback_type='pr_norm')
                if info:
                    counters['krylov_failures'] += 1
            counters['outer_linear_seconds'] += time.perf_counter()-linear_started
            directional = float(grad_envelope @ direction)
            if not np.isfinite(directional) or directional >= -1e-10*np.linalg.norm(grad_envelope)*np.linalg.norm(direction):
                direction = -r
                directional = float(grad_envelope @ direction)
                counters['outer_gradient_fallbacks'] += 1
            accepted = False
            alpha = 1.0
            for _ in range(20):
                trial = point(x+alpha*direction, current['y'], inner_tolerance=current['inner_tolerance'])
                allowance = 1e-14*max(1.0, abs(current['envelope']))
                if trial['inner_residual'] <= max(1e-8, feas_tol*0.1, trial['inner_tolerance']*1.1) and trial['envelope'] <= current['envelope'] + 1e-4*alpha*directional + allowance:
                    x, current, accepted = x+alpha*direction, trial, True
                    break
                alpha *= 0.5
                counters['outer_backtracks'] += 1
            if not accepted:
                counters['outer_gradient_fallbacks'] += 1
                x = current['x'].copy()
                current = point(x, current['y'], inner_tolerance=current['inner_tolerance'])
    except _Timeout:
        status = 'time_limit'
    if last is None:
        # A very short deadline can expire before the first composite prox.
        last = dict(x=x, y=y, inner_residual=float('inf'))
    counters.update(seconds=time.perf_counter()-started, tau=float(tau),
                    inner_residual=float(last['inner_residual']), mode=mode,
                    inexact_forcing=inexact,
                    mean_inner_support=counters['inner_support_sum']/max(1,counters['inner_structured_solves']),
                    mean_active_constraints=counters['active_constraint_sum']/max(1,counters['inner_iterations']))
    return dict(x=last['x'], y=last['y'], iterations=iteration,
                status=status, diagnostics=counters, history=history)
