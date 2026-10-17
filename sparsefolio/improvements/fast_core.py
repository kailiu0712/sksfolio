"""Faster scalar roots with the same perspective prox and value definitions.

The prox first solves in c=sqrt(2*theta), where each fixed active set has
an explicit root. It validates that root with the original capped formula;
the original safeguarded solver handles cases where asset caps change it.
"""
import numpy as np
from numba import njit
from ..core import _at_theta, prox as original_prox


@njit(cache=True)
def prox(v, tau, gamma, u, k):
    a = tau * gamma
    x,z,d,h,den = _at_theta(v,a,u,0.0)
    if z.sum() <= k:
        return x,z,0.0,d,h,0.0
    sq = np.sqrt(a)
    top, bottom, upper = 0.0, float(k), 0.0
    for i in range(len(v)):
        if v[i] > 0:
            top += v[i]*sq[i]
            bottom += a[i]
            upper = max(upper,v[i]/sq[i])
    c = top / bottom
    lower = 0.0
    for iteration in range(60):
        mass, top, bottom = 0.0, 0.0, float(k)
        for i in range(len(v)):
            if v[i] <= c*sq[i]:
                continue
            if v[i] >= c*(sq[i]+1.0/sq[i]):
                mass += 1.0
                bottom -= 1.0
            else:
                term = v[i]*sq[i]
                mass += term/c-a[i]
                top += term
                bottom += a[i]
        if abs(mass-k) <= 2e-12*max(1.0,k):
            break
        if mass > k:
            lower = c
        else:
            upper = c
        next_c = top / bottom if bottom>0 else -1.0
        if next_c <= lower or next_c >= upper:
            next_c = .5*(lower+upper)
        c = next_c
    theta = .5*c*c
    x,z,d,h,den = _at_theta(v,a,u,theta)
    if abs(z.sum()-k) <= 3e-11*max(1.0,k):
        return x,z,theta,d,h,den
    return original_prox(v,tau,gamma,u,k)


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
