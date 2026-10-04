"""A collection of functions that deal with missing values"""

import numpy as np
from functools import reduce


def miss_tensordot(X, u, missX=None):
    # Equivalent to np.einsum("i...,i...->...", X, u), but X with missing values at missX
    Xdim = X.shape
    assert Xdim[0] == u.shape[0]
    if missX is None:
        missX = np.isnan(X)
    X = X.reshape(Xdim[0], -1)
    missX = missX.reshape(Xdim[0], -1)
    w = np.zeros((X.shape[1],))
    for i in range(X.shape[1]):
        m = np.where(~missX[:, i])[0]
        if len(m) > 0:
            w[i] = X[m, i].T @ u[m] / len(m) * Xdim[0]
    return w.reshape(Xdim[1:])


def miss_mmodedot(X, facs, missX=None):
    # Equivalent to multi_mode_dot(X, fac, range(1, X.ndim)), but X with missing values at missX
    # facs ~= [ff[:, a] for ff in self.X_factors[1:]]
    Xdim = X.shape
    assert all([(Xdim[i + 1], ff.shape[0]) for (i, ff) in enumerate(facs)])
    if missX is None:
        missX = np.isnan(X)
    X = X.reshape(Xdim[0], -1)
    missX = missX.reshape(Xdim[0], -1)
    t = np.empty((Xdim[0],))
    t.fill(np.nan)
    wkron = reduce(np.kron, facs)
    Wdim = wkron.shape[0]
    for i in range(Xdim[0]):
        m = np.where(~missX[i, :])[0]
        if len(m) > 0:
            t[i] = X[i, m] @ wkron[m] / len(m) * Wdim
    return t
