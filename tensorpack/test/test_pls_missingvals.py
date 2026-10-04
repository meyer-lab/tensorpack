"""Test the functions for dealing with missing values"""

import numpy as np
import pytest
from numpy.linalg import norm
from tensorly.tenalg import (
    multi_mode_dot,  # ty: ignore[unresolved-import]  # backend-dispatched at runtime
)

from tensorpack.pls.missingvals import miss_mmodedot, miss_tensordot
from tensorpack.pls.synthetic import import_synthetic
from tensorpack.pls.tpls import calcR2X, tPLS


def test_miss_tensordot():
    # Test it is equivalent to np.einsum("i...,i...->...", X, u)
    X = np.random.rand(10, 5, 4, 3)
    X[np.random.rand(*X.shape) < 0.1] = np.nan
    missX = np.isnan(X)
    u = np.random.rand(10)
    w = miss_tensordot(X, u, missX.reshape(X.shape[0], -1))
    w2 = np.einsum("i...,i...->...", X, u)
    assert np.allclose(w * ~np.isnan(w2), np.nan_to_num(w2))

    total_error = 0
    for _ in range(10):
        X = np.random.rand(20, 1) @ np.random.rand(8, 1).T
        u = np.random.rand(20)
        w = X.T @ u
        X[np.random.rand(*X.shape) < 0.2] = np.nan
        w1 = miss_tensordot(X, u)
        w2 = np.nan_to_num(X.T) @ u
        assert norm(w - w1) / norm(w) < norm(w - w2) / norm(w) + 0.01
        total_error += norm(w - w1) / norm(w)
    assert total_error < 1.2


def test_miss_mmodedot():
    # Test it is equivalent to multi_mode_dot(X, fac, range(1, X.ndim))
    total_error = 0
    for _ in range(10):
        X = np.random.rand(10, 9, 8, 7)
        facs = [np.random.rand(lf) for lf in X.shape[1:]]
        t = multi_mode_dot(X, facs, range(1, X.ndim))
        X[np.random.rand(*X.shape) < 0.1] = np.nan
        missX = np.isnan(X)
        t1 = miss_mmodedot(X, facs, missX)
        t2 = multi_mode_dot(np.nan_to_num(X), facs, range(1, X.ndim))
        assert norm(t - t1) / norm(t) < norm(t - t2) / norm(t) + 0.01
        total_error += norm(t - t1) / norm(t)
    assert total_error < 1.2


def test_miss_mmodedot_completeMissSample():
    """Test that with complete missing sample slides it gives np.nan"""
    X = np.random.rand(10, 9, 8, 7)
    X[7:, :, :, :] = np.nan
    facs = [np.random.rand(l) for l in X.shape[1:]]
    t = miss_mmodedot(X, facs)
    assert np.all(np.isnan(t[7:]))
    assert np.all(~np.isnan(t[:7]))


@pytest.mark.parametrize("Xshape", [(10, 9, 8), (10, 9, 8, 7), (10, 9, 8, 7, 6)])
def test_miss_X_synthetic(Xshape):
    X, Y, _ = import_synthetic(Xshape, 4, 1, seed=np.random.randint(1000))
    tpls = tPLS(1)
    tpls.fit(X, Y)
    X[np.random.rand(*X.shape) < 0.1] = np.nan
    tpls1 = tPLS(1)
    tpls1.fit(X, Y)
    for i in range(X.ndim):
        fac = tpls.X_factors[i]
        fac1 = tpls1.X_factors[i]
        assert (norm(fac - fac1) / norm(fac)) < 0.2
    for i in range(Y.ndim):
        fac = tpls.Y_factors[i]
        fac1 = tpls1.Y_factors[i]
        assert (norm(fac - fac1) / norm(fac)) < 0.01


def test_miss_X_transform():
    X = np.random.rand(10, 7, 6, 5)
    Y = np.random.rand(10, 4)
    X[np.random.rand(*X.shape) < 0.2] = np.nan
    tpls = tPLS(7)
    tpls.fit(X, Y)
    assert np.all(np.diff(tpls.R2X) >= 0.0)
    assert np.all(np.diff(tpls.R2Y) >= 0.0)
    Xsc, Ysc = tpls.transform(X, Y)
    assert np.allclose(tpls.X_factors[0], Xsc)
    assert np.allclose(tpls.Y_factors[0], Ysc)


def test_miss_X_imputation():
    """Test that PLSR can impute missing values"""
    X, Y, _ = import_synthetic((10, 9, 8, 7), 4, 3, seed=np.random.randint(1000))
    Xmiss = X.copy()
    missPos = np.random.rand(*X.shape) < 0.25
    Xmiss[missPos] = np.nan
    tpls = tPLS(3)
    tpls.fit(Xmiss, Y)
    assert calcR2X(X[missPos], tpls.X_reconstructed()[missPos]) > 0.8
