"""Independent finite differences and redundant-row checks for Newton prox."""
import numpy as np
import pytest
from sksfolio.relaxation.pava.partial_sort import prox
from sksfolio.relaxation.fista.semismooth import pava_jacobian_parts
from sksfolio.relaxation.fista import LinearConstraintProx


@pytest.mark.parametrize('gamma', [0.01,0.5,2.,100.])
@pytest.mark.parametrize('k', [1,3,20])
def test_jacobian_finite_difference(gamma,k):
    rng=np.random.default_rng(432+k)
    for _ in range(20):
        values=rng.normal(size=20)*3
        direction=rng.normal(size=20)
        point=prox(values,gamma,k)
        diagonal,pool,beta=pava_jacobian_parts(values,point,gamma,k)
        product=diagonal*direction-beta*pool*np.sum(direction[pool])
        finite=(prox(values+1e-6*direction,gamma,k)-prox(values-1e-6*direction,gamma,k))/2e-6
        np.testing.assert_allclose(product,finite,atol=2e-7,rtol=2e-6)


def test_newton_redundant_equalities_and_sector_intervals():
    rng=np.random.default_rng(3)
    matrix=np.array([[1,1,1,1],[2,2,2,2],[1,1,0,0],[0,0,1,1]],dtype=float)
    lower=np.array([1.,2.,.2,.2])
    upper=np.array([1.,2.,.8,.8])
    values=rng.normal(size=4)
    reference=LinearConstraintProx(matrix,lower,upper,2,tolerance=1e-9).solve(values,2.)
    result=LinearConstraintProx(matrix,lower,upper,2,tolerance=1e-9,semismooth_newton=True).solve(values,2.)
    assert result.converged
    assert result.constraint_violation < 1e-8
    np.testing.assert_allclose(result.x,reference.x,atol=1e-7)
