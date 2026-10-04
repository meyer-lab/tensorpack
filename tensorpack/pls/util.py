import numpy as np
from numpy.linalg import norm
from tensorly import fold
from tensorly.tenalg import khatri_rao


def calcR2X(X, Xhat):
    if (Xhat.ndim == 2) and (X.ndim == 1):
        X = X.reshape(-1, 1)
    assert X.shape == Xhat.shape
    mask = np.isfinite(X)
    xIn = np.nan_to_num(X)
    top = norm(Xhat * mask - xIn) ** 2.0
    bottom = norm(xIn) ** 2.0
    return 1 - top / bottom


def factors_to_tensor(factors):
    full_tensor = factors[0] @ khatri_rao(factors, skip_matrix=0).T
    return fold(full_tensor, 0, [ff.shape[0] for ff in factors])
