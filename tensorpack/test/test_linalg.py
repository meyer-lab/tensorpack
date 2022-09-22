import numpy as np
from ..linalg import mlstsq, lstsq_

def test_regular_lstsq():
    A = np.random.rand(10, 8)
    B = np.random.rand(10, 5)
    x = np.linalg.lstsq(A, B, rcond=-1)[0]
    assert np.all(x == lstsq_(A, B))
    xp = mlstsq(A, B, nonneg=True)
    assert np.all(xp >= 0)
    assert np.sum(np.abs((A @ xp - B) / B) < 1.0) > 30