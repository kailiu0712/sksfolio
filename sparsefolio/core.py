"""Independent root-relaxation primitives and a shared accuracy assessment.

All quantities use x notation.  The only supported buy-in lower bound is zero.
The cardinality budget is relaxed through the perspective function, not removed.
"""
from dataclasses import dataclass, field
import numpy as np
from numba import njit


@dataclass
class Problem:
    B: np.ndarray
    c: np.ndarray
    gamma: np.ndarray
    u: np.ndarray
    k: int
    K: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    raw_K: np.ndarray | None = None
    raw_lo: np.ndarray | None = None
    raw_hi: np.ndarray | None = None
    row_scale: np.ndarray | None = None
    anchor: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def n(self):
        return len(self.c)


@njit(cache=True)
def _at_theta(v, a, u, theta):
    n = len(v)
    x, z = np.zeros(n), np.zeros(n)
    d, h = np.zeros(n), np.zeros(n)
    denom = 0.0
    for i in range(n):
        if theta == 0.0:
            x[i] = min(u[i], max(0.0, v[i] / (1.0 + a[i])))
            z[i] = 1.0 if x[i] > 0 else 0.0
            if 0.0 < x[i] < u[i]:
                d[i] = 1.0 / (1.0 + a[i])
            continue
        r = min(u[i], np.sqrt(2.0 * theta / a[i]))
        shift = a[i] * r / 2.0 + theta / r
        if v[i] <= shift:
            continue
        if v[i] < shift + r:
            x[i] = v[i] - shift
            z[i] = x[i] / r
            d[i], h[i] = 1.0, 1.0 / r
            denom += 1.0 / (r * r)
            if r < u[i]:
                denom += x[i] / (a[i] * r**3)
        else:
            x[i] = min(u[i], max(r, v[i] / (1.0 + a[i])))
            z[i] = 1.0
            if x[i] < u[i]:
                d[i] = 1.0 / (1.0 + a[i])
    return x, z, d, h, denom


@njit(cache=True)
def prox(v, tau, gamma, u, k):
    """Return x,z,theta,d,h,denom with J=diag(d)-h h.T/denom.

    A safeguarded scalar Newton solve enforces sum(z)<=k.
    The final root bracket supplies a deterministic fallback near breakpoints.
    """
    a = tau * gamma
    x, z, d, h, den = _at_theta(v, a, u, 0.0)
    if z.sum() <= k:
        return x, z, 0.0, d, h, 0.0
    lower, upper = 0.0, 1.0
    while True:
        x, z, d, h, den = _at_theta(v, a, u, upper)
        if z.sum() <= k:
            break
        upper *= 2.0
    theta = 0.5 * upper
    for _ in range(100):
        x, z, d, h, den = _at_theta(v, a, u, theta)
        residual = z.sum() - k
        if abs(residual) <= 2e-12 * max(1.0, k):
            break
        if residual > 0:
            lower = theta
        else:
            upper = theta
        candidate = theta + residual / den if den > 0.0 else -1.0
        if candidate <= lower or candidate >= upper:
            candidate = 0.5 * (lower + upper)
        theta = candidate
    return x, z, theta, d, h, den


@njit(cache=True)
def value_and_z(x,gamma,u,k):
    n=len(x)
    xx=np.maximum(x,0.0)
    lower=xx/u
    if np.max(xx-u)>1e-9 or lower.sum()>k+1e-13*max(1.0,k):
        return np.inf,lower
    z=np.zeros(n)
    if np.sum(xx>0.0)<=k:
        value=0.0
        for i in range(n):
            if xx[i]>0:
                z[i]=1.0
                value+=.5*gamma[i]*xx[i]*xx[i]
        return value,z
    w=xx*np.sqrt(gamma)
    lo,hi=0.0,max(1.0,w.sum()/k)
    # Floors z>=x/u can keep the mass above k even after unclipped
    # water-filling would be below k. Establish an actual upper bracket.
    if abs(lower.sum()-k)<=1e-13*max(1.0,k):
        z=lower.copy()
        value=0.0
        for i in range(n):
            if xx[i]>0:
                value+=.5*gamma[i]*xx[i]*xx[i]/z[i]
        return value,z
    while np.minimum(1.0,np.maximum(lower,w/hi)).sum()>k:
        hi*=2.0
    eta=w.sum()/k
    for iteration in range(60):
        mass,top,bottom=0.0,0.0,float(k)
        for i in range(n):
            candidate=w[i]/eta
            if candidate>=1.0:
                z[i]=1.0
                bottom-=1.0
            elif candidate<=lower[i]:
                z[i]=lower[i]
                bottom-=lower[i]
            else:
                z[i]=candidate
                top+=w[i]
            mass+=z[i]
        if abs(mass-k)<=2e-12*max(1.0,k):
            break
        if mass>k:
            lo=eta
        else:
            hi=eta
        nxt=top/bottom if bottom>0 else -1.0
        if nxt<=lo or nxt>=hi:
            nxt=.5*(lo+hi)
        eta=nxt
    if abs(z.sum()-k)>3e-11*max(1.0,k):
        for iteration in range(100):
            eta=.5*(lo+hi)
            z=np.minimum(1.0,np.maximum(lower,w/eta))
            mass=z.sum()
            if abs(mass-k)<=2e-12*max(1.0,k):
                break
            if mass>k:
                lo=eta
            else:
                hi=eta
    value=0.0
    for i in range(n):
        if xx[i]>0:
            value+=.5*gamma[i]*xx[i]*xx[i]/z[i]
    return value,z


def evaluate(p, x, y):
    """Evaluate original-unit feasibility and a residual-qualified dual gap.

    The lower bound follows from a quadratic supporting hyperplane and the
    conjugate of the perspective envelope, the sum of the k largest benefits.
    A not-yet-feasible x does not give a rigorous primal upper bound.
    Thus feasibility and gap are always assessed separately.
    """
    y = np.asarray(y).copy()
    y[np.isneginf(p.lo)] = np.maximum(y[np.isneginf(p.lo)], 0.0)
    y[np.isposinf(p.hi)] = np.minimum(y[np.isposinf(p.hi)], 0.0)
    bx = p.B @ x
    v = p.c + p.B.T @ bx + p.K.T @ y
    t = np.clip(-v / p.gamma, 0.0, p.u)
    benefit = -v * t - 0.5 * p.gamma * t * t
    largest = np.partition(benefit, max(0, p.n - p.k))[max(0, p.n - p.k):]
    support = float(y[y > 0] @ p.hi[y > 0] + y[y < 0] @ p.lo[y < 0])
    lb = float(-0.5 * (bx @ bx) - support - largest.sum())
    gv, z = value_and_z(x, p.gamma, p.u, p.k)
    obj = float(0.5 * (bx @ bx) + p.c @ x + gv)
    K = p.raw_K if p.raw_K is not None else p.K
    lo = p.raw_lo if p.raw_lo is not None else p.lo
    hi = p.raw_hi if p.raw_hi is not None else p.hi
    ax = K @ x
    feas = max(float(np.max(np.maximum(lo - ax, ax - hi))),
               float(np.max(-x)), float(np.max(x - p.u)),
               float(np.sum(np.maximum(x, 0) / p.u) - p.k), 0.0)
    gap = obj - lb
    relative = abs(gap) / max(abs(obj), abs(lb), 1e-8) if np.isfinite(obj) else np.inf
    return dict(objective=obj, lower_bound=lb, dual_gap=gap, rel_gap=relative,
                feasibility=feas, z_sum=float(z.sum()))


def warmup():
    v = np.array([0.5, 0.8, -0.2])
    gamma, u = np.ones(3), np.ones(3)
    prox(v, 0.5, gamma, u, 1)
    value_and_z(np.array([0.1, 0.2, 0.3]), gamma, u, 1)
