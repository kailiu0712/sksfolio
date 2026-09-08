from types import SimpleNamespace
import numpy as np
from scipy import sparse
from sksfolio.bnb.cardinality_presolve import cardinality_certificate


def problem(matrix,lower,k,upper=None):
    return SimpleNamespace(C=sparse.csr_matrix(matrix),lower=np.array(lower),
                           upper=np.full(len(lower),np.inf) if upper is None else np.array(upper),k=k)


def test_disjoint_sector_proof():
    p=problem([[1,1,1,1,1,1],[1,1,0,0,0,0],[0,0,1,1,0,0],[0,0,0,0,1,1]],
              [1,.2,.2,.2],2)
    certificate=cardinality_certificate(p)
    assert certificate['minimum_assets']==3
    assert [entry['row'] for entry in certificate['rows']]==[1,2,3]
    p.k=3
    assert cardinality_certificate(p) is None


def test_overlapping_rows_are_not_counted_twice():
    assert cardinality_certificate(problem([[1,1,0],[0,1,1]],[.2,.2],1)) is None


def test_negative_upper_and_forbidden_assets():
    p=problem([[-1,-1,0,0],[0,0,1,1]],[-np.inf,.2],1,[-.2,np.inf])
    assert cardinality_certificate(p)['minimum_assets']==2
    p.k=2
    assert cardinality_certificate(p) is None
    assert cardinality_certificate(p,[0,1])['kind']=='positive_row_without_available_support'


def test_nonpositive_lower_gives_no_support_requirement():
    assert cardinality_certificate(problem(np.eye(4),[0,-1,0,-np.inf],1)) is None
