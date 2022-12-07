from ..tpls import *
from tensorly.random import random_cp

def test_random_tensor():
    X = np.random.rand(9,8,7,6)
    Y = np.random.rand(9,4)
    oldR2X = -np.inf
    for r in [2,4,6,8]:
        cp = tPLS(r)
        cp.fit(X, Y, max_iter=200)
        assert cp.centered_R2X() >= oldR2X, "R2X is not increasing with more components"
        oldR2X = cp.centered_R2X()
        covs = [np.cov(cp.X_factors[0][:, cc], cp.Y_factors[0][:, cc])[0, 1] for cc in range(r)]
        #assert all([covs[ii] >= covs[ii+1] for ii in range(len(covs)-1)]), "Covariance not in descending order"

def test_synthetic_tensor():
    X_cp = random_cp((9,8,7,6), 5)
    Y_cp = random_cp((9,5), 5)
    Y_cp[1][0] = X_cp[1][0]

    oldR2X = -np.inf
    for r in [2,4,6,8]:
        cp = tPLS(r)
        cp.fit(X_cp.to_tensor(), Y_cp.to_tensor(), max_iter=200)
        assert cp.centered_R2X() >= 0.6, "R2X is too small for a synthetic tensor"
        assert cp.centered_R2X() >= oldR2X, "R2X is not increasing with more components"
        print(r, cp.centered_R2X(), cp.centered_R2Y())
        oldR2X = cp.centered_R2X()
        covs = [np.cov(cp.X_factors[0][:, cc], cp.Y_factors[0][:, cc])[0, 1] for cc in range(r)]
        assert all([covs[ii] >= covs[ii+1] for ii in range(len(covs)-1)]), "Covariance not in descending order"

