"""Experimental structured primal-dual IPM for the perspective relaxation.

The explicit variables x,z satisfy x>=0, 0<=z<=1, x<=u*z, sum(z)<=k.
The perspective objective is .5*sum(gamma*x*x/z).  A damped central-path Newton
step eliminates inequality slacks and duals, inverts independent 2-by-2 asset
blocks, and applies Woodbury to factors, exposures, and sum(z).  The supplied
feasible anchor is strictified when available; otherwise the start is uniform
and infeasible.  No reference solution or external optimizer enters the solve.

This prototype loses numerical accuracy near the barrier boundary on some
large exposure cases.  A success status is returned only if the unchanged
external certificate actually passes.  It is not a validated benchmark winner.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import LinearOperator, gmres

try:
    from ..core import evaluate
except ImportError:
    from ..core import evaluate


class _TimeLimit(Exception):
    pass


def _step_length(value, direction, fraction=1.0):
    decreasing = direction < 0
    if not np.any(decreasing):
        return 1.0
    return min(1.0, fraction*float(np.min(-value[decreasing]/direction[decreasing])))


def solve(problem, mode='ipm', time_limit=15.0, tol=1e-6,
          feas_tol=1e-7, max_iter=100):
    """Solve from a strictified input anchor, or a uniform infeasible start.

    The returned certificate uses exactly core.evaluate and the same requested
    original-unit feasibility and relative-gap tolerances as the v1 solvers.
    """
    if mode not in ('ipm', 'structured_ipm'):
        raise ValueError('mode must be ipm or structured_ipm')
    start=time.perf_counter()
    deadline=start+float(time_limit)
    p=problem
    n=p.n
    B=np.ascontiguousarray(p.B)
    equal=np.flatnonzero(p.lo==p.hi)
    exposure=np.flatnonzero(p.lo!=p.hi)
    E=np.ascontiguousarray(p.K[equal])
    target=p.lo[equal]
    A=np.ascontiguousarray(p.K[exposure])
    lower=np.flatnonzero(np.isfinite(p.lo[exposure]))
    upper=np.flatnonzero(np.isfinite(p.hi[exposure]))
    q=len(B)
    m=len(A)
    ne=len(E)
    nl,nu=len(lower),len(upper)
    dl=slice(4*n+1,4*n+1+nl)
    du=slice(4*n+1+nl,4*n+1+nl+nu)
    total_constraints=4*n+1+nl+nu
    b=np.r_[np.zeros(2*n),np.ones(n),np.zeros(n),float(p.k),
            -p.lo[exposure][lower],p.hi[exposure][upper]]
    Q=np.ascontiguousarray(np.vstack((B,A)))
    x=np.full(n,1.0/n)
    if p.anchor is not None:
        mixing=0.05
        for _ in range(40):
            candidate=(1-mixing)*p.anchor+mixing*x
            av=A@candidate
            if np.all(av[lower]>p.lo[exposure][lower]) and np.all(av[upper]<p.hi[exposure][upper]):
                x=candidate
                break
            mixing*=0.5
    z=x/p.u+(float(p.k)-np.sum(x/p.u))/(2*n)
    if z.sum()>=p.k:
        z[:]=float(p.k)*0.75/n
    if np.any(z<=0):
        raise ValueError('The perspective model requires k>0 and u>0')
    eq_dual=np.zeros(ne)
    counters=dict(mode=mode,initialization='strictified_input_anchor' if p.anchor is not None else 'uniform_infeasible_start',
                  factorizations=0,factorization_seconds=0.0,linear_seconds=0.0,
                  schur_dimension=q+m+1,asset_blocks=n,regularizations=0,
                  certificates=0,refinement_iterations=0)
    history=[]
    best=None
    best_merit=np.inf
    status='iteration_limit'

    def check_time():
        if time.perf_counter()>=deadline:
            raise _TimeLimit

    def g_action(dx,dz):
        ax=A@dx
        return np.r_[-dx,-dz,dz,dx-p.u*dz,dz.sum(),-ax[lower],ax[upper]]

    def g_transpose(v):
        coefficients=np.zeros(m)
        coefficients[lower]-=v[dl]
        coefficients[upper]+=v[du]
        rx=-v[:n]+v[3*n:4*n]+A.T@coefficients
        rz=-v[n:2*n]+v[2*n:3*n]-p.u*v[3*n:4*n]+v[4*n]
        return rx,rz

    s=np.maximum(b-g_action(x,z),1e-8)
    dual=(0.1/n)/s

    def assess(iteration,mu):
        nonlocal best,best_merit
        y=np.zeros(len(p.lo))
        y[equal]=eq_dual
        coefficients=np.zeros(m)
        coefficients[lower]-=dual[dl]
        coefficients[upper]+=dual[du]
        y[exposure]=coefficients
        metrics=evaluate(p,x,y)
        merit=max(metrics['rel_gap']/tol,metrics['feasibility']/feas_tol)
        counters['certificates']+=1
        history.append(dict(iteration=iteration,seconds=time.perf_counter()-start,
                            complementarity=mu,**metrics))
        if best is None or merit<best_merit:
            best=(x.copy(),y,metrics)
            best_merit=merit
        return merit<=1.0

    try:
        for iteration in range(max_iter+1):
            check_time()
            mu=float(s@dual/total_constraints)
            if assess(iteration,mu):
                status='converged'
                break
            if iteration==max_iter:
                break
            ratio=x/z
            bx=B@x
            gx=B.T@bx+p.c+p.gamma*ratio
            gz=-0.5*p.gamma*ratio*ratio
            gtx,gtz=g_transpose(dual)
            rd_x=gx+gtx+E.T@eq_dual
            rd_z=gz+gtz
            rp=g_action(x,z)+s-b
            req=E@x-target
            weight=dual/s
            wx,wz,wu,wcap=(weight[:n],weight[n:2*n],weight[2*n:3*n],weight[3*n:4*n])
            perspective_curvature=p.gamma/z
            aa=perspective_curvature+wx+wcap
            ab=-perspective_curvature*ratio-p.u*wcap
            dd=perspective_curvature*ratio*ratio+wz+wu+p.u*p.u*wcap
            # Expanded determinant avoids cancellation in the rank-one
            # perspective Hessian when barrier curvature becomes small.
            determinant=(perspective_curvature*(wz+wu+(p.u-ratio)**2*wcap+ratio*ratio*wx)
                         +(wx+wcap)*(wz+wu)+p.u*p.u*wx*wcap)
            inv_x=dd/determinant
            inv_cross=-ab/determinant
            inv_z=aa/determinant
            exp_weight=np.zeros(m)
            exp_weight[lower]+=weight[dl]
            exp_weight[upper]+=weight[du]
            ux=Q*np.sqrt(np.r_[np.ones(q),exp_weight])[:,None]
            card_root=np.sqrt(weight[4*n])
            linear_start=time.perf_counter()
            dim=q+m+1
            schur=np.empty((dim,dim),order='F')
            schur[:-1,:-1]=(ux*inv_x)@ux.T
            schur[:-1,-1]=card_root*(ux@inv_cross)
            schur[-1,:-1]=schur[:-1,-1]
            schur[-1,-1]=weight[4*n]*inv_z.sum()
            schur.flat[::dim+1]+=1.0
            scale=np.sqrt(np.diag(schur))
            schur/=scale[:,None]
            schur/=scale[None,:]
            factor_start=time.perf_counter()
            try:
                chol=cho_factor(schur,lower=True,check_finite=False,overwrite_a=True)
            except np.linalg.LinAlgError:
                # Rare late-iteration roundoff in the scaled Schur complement.
                schur.flat[::dim+1]+=1e-12
                chol=cho_factor(schur,lower=True,check_finite=False,overwrite_a=True)
                counters['regularizations']+=1
            counters['factorization_seconds']+=time.perf_counter()-factor_start
            counters['factorizations']+=1

            def h_inverse(rx,rz):
                vector=rx.ndim==1
                if vector:
                    rx,rz=rx[:,None],rz[:,None]
                dx=inv_x[:,None]*rx+inv_cross[:,None]*rz
                dz=inv_cross[:,None]*rx+inv_z[:,None]*rz
                ur=np.vstack((ux@dx,card_root*np.sum(dz,axis=0,keepdims=True)))
                solution=cho_solve(chol,ur/scale[:,None],check_finite=False)/scale[:,None]
                xt=ux.T@solution[:-1]
                zt=card_root*solution[-1]
                dx-=inv_x[:,None]*xt+inv_cross[:,None]*zt
                dz-=inv_cross[:,None]*xt+inv_z[:,None]*zt
                return (dx[:,0],dz[:,0]) if vector else (dx,dz)

            if ne:
                hx,hz=h_inverse(E.T,np.zeros((n,ne)))
                equality_schur=E@hx
                equality_chol=cho_factor(equality_schur,lower=True,check_finite=False)

            def direction(rc):
                correction=rc/s-weight*rp
                cx,cz=g_transpose(correction)
                rhsx,rhsz=-rd_x+cx,-rd_z+cz
                dx,dz=h_inverse(rhsx,rhsz)
                if ne:
                    deq=cho_solve(equality_chol,E@dx+req,check_finite=False)
                    dx-=hx@deq
                    dz-=hz@deq
                else:
                    deq=np.zeros(0)
                def action(vector):
                    dx,dz,de=vector[:n],vector[n:2*n],vector[2*n:]
                    return np.r_[aa*dx+ab*dz+B.T@(B@dx)+A.T@(exp_weight*(A@dx))+E.T@de,
                                 ab*dx+dd*dz+weight[4*n]*dz.sum(),E@dx]
                def precondition(vector):
                    sx,sz=h_inverse(vector[:n],vector[n:2*n])
                    if ne:
                        se=cho_solve(equality_chol,E@sx-vector[2*n:],check_finite=False)
                        sx-=hx@se
                        sz-=hz@se
                    else:
                        se=np.zeros(0)
                    return np.r_[sx,sz,se]
                rhs=np.r_[rhsx,rhsz,-req]
                initial=np.r_[dx,dz,deq]
                linear_residual=rhs-action(initial)
                if np.linalg.norm(linear_residual)>max(1e-11,1e-8*np.linalg.norm(rhs)):
                    def callback(_):
                        counters['refinement_iterations']+=1
                        check_time()
                    operator=LinearOperator((2*n+ne,2*n+ne),matvec=action,dtype=float)
                    preconditioner=LinearOperator(operator.shape,matvec=precondition,dtype=float)
                    refined,info=gmres(operator,rhs,x0=initial,M=preconditioner,
                                       rtol=1e-10,atol=1e-12,restart=30,maxiter=3,
                                       callback=callback,callback_type='pr_norm')
                    if np.linalg.norm(rhs-action(refined))<np.linalg.norm(linear_residual):
                        dx,dz,deq=refined[:n],refined[n:2*n],refined[2*n:]
                ds=-rp-g_action(dx,dz)
                ddual=-(rc+dual*ds)/s
                return dx,dz,ds,ddual,deq

            target_mu=0.15*mu
            rc=s*dual-target_mu
            dx,dz,ds,ddual,deq=direction(rc)
            counters['linear_seconds']+=time.perf_counter()-linear_start
            alpha=min(_step_length(s,ds,0.995),_step_length(z,dz,0.995),
                      _step_length(dual,ddual,0.995))
            old_merit=rd_x@rd_x+rd_z@rd_z+rp@rp+req@req+(rc@rc)/mu
            accepted=False
            for _ in range(24):
                xt,zt=x+alpha*dx,z+alpha*dz
                st,dt=s+alpha*ds,dual+alpha*ddual
                et=eq_dual+alpha*deq
                rt=xt/zt
                dgx,dgz=g_transpose(dt)
                rdx=B.T@(B@xt)+p.c+p.gamma*rt+dgx+E.T@et
                rdz=-0.5*p.gamma*rt*rt+dgz
                rpt=g_action(xt,zt)+st-b
                eqt=E@xt-target
                rct=st*dt-target_mu
                merit=rdx@rdx+rdz@rdz+rpt@rpt+eqt@eqt+(rct@rct)/mu
                if merit<=(1-1e-3*alpha)*old_merit:
                    x,z,s,dual,eq_dual=xt,zt,st,dt,et
                    accepted=True
                    break
                alpha*=0.5
            if not accepted:
                status='line_search_failure'
                break
            if not (np.all(np.isfinite(x)) and np.all(np.isfinite(z))):
                status='numerical_failure'
                break
    except _TimeLimit:
        status='time_limit'
    except np.linalg.LinAlgError:
        status='linear_failure'
    if best is None:
        assess(0,float(s@dual/total_constraints))
    if best_merit<=1.0:
        status='converged'
    xx,yy,metrics=best
    counters.update(seconds=time.perf_counter()-start,selected_merit=best_merit,
                    primal_dual_inequalities=total_constraints)
    return dict(x=xx,y=yy,iterations=iteration,status=status,
                diagnostics=counters,metrics=metrics,history=history,
                seconds=counters['seconds'])
