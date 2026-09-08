"""Sound disjoint-support cardinality infeasibility certificates.

For x >= 0 and x <= z, a row with positive lower bound needs a positive
coefficient variable selected. Disjoint such supports need distinct assets.
Negative upper bounds supply the same condition after negating the row.
This uses the original inequalities, without treating dependent rows as
redundant. A missed certificate simply leaves the original solver running.
"""
from __future__ import annotations
import numpy as np
from scipy import sparse


def cardinality_certificate(instance, forbidden_assets=(), tolerance=1e-10):
    matrix=sparse.csr_matrix(instance.C,copy=True)
    matrix.sum_duplicates()
    candidates=[]
    forbidden=set(map(int,forbidden_assets))
    for row in range(matrix.shape[0]):
        start,stop=matrix.indptr[row:row+2]
        columns=matrix.indices[start:stop]
        values=matrix.data[start:stop]
        for sign,bound in ((1.,float(instance.lower[row])),(-1.,-float(instance.upper[row]))):
            if not np.isfinite(bound) or bound <= tolerance:
                continue
            support=frozenset(int(i) for i,v in zip(columns,sign*values) if v>0 and int(i) not in forbidden)
            if not support:
                return dict(kind='positive_row_without_available_support',rows=[row],minimum_assets=1,k=int(instance.k))
            candidates.append((len(support),row,sign,support))
    candidates.sort(key=lambda item:(item[0],item[1],item[2]))
    used=set()
    selected=[]
    for _,row,sign,support in candidates:
        if used.isdisjoint(support):
            used.update(support)
            selected.append(dict(row=row,sign=sign,support=sorted(support)))
            if len(selected)>instance.k:
                return dict(kind='disjoint_positive_row_supports',rows=selected,
                            minimum_assets=len(selected),k=int(instance.k))
    return None
