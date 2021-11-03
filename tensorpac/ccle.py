import numpy as np
from scipy.optimize import minimize
import tensorly as tl


tl.set_backend('numpy')


def calcR2X(tFac, tIn=None, mIn=None):
    """ Calculate R2X. Optionally it can be calculated for only the tensor or matrix. """
    assert (tIn is not None) or (mIn is not None)

    vTop = 0.0
    vBottom = 0.0

    if tIn is not None:
        tMask = np.isfinite(tIn)
        vTop += np.sum(np.square(tl.cp_to_tensor(tFac) * tMask - np.nan_to_num(tIn)))
        vBottom += np.sum(np.square(np.nan_to_num(tIn)))
    if mIn is not None:
        mMask = np.isfinite(mIn)
        vTop += np.sum(np.square(tl.cp_to_tensor(tFac.mFactor) * mMask - np.nan_to_num(mIn)))
        vBottom += np.sum(np.square(np.nan_to_num(mIn)))

    return 1.0 - vTop / vBottom


def cp_to_vec(tFac):
    return np.concatenate([tFac.factors[i].flatten() for i in range(2)])


def buildTensors(pIn, tensor, matrix, r, cost=False):
    """ Use parameter vector to build kruskal tensors. """
    assert tensor.shape[0] == matrix.shape[0]
    nN = np.cumsum(np.array(tensor.shape) * r)
    A = np.reshape(pIn[:nN[0]], (tensor.shape[0], r))
    B = np.reshape(pIn[nN[0]:nN[1]], (tensor.shape[1], r))

    kr = tl.tenalg.khatri_rao([A, B])
    unfold = tl.unfold(tensor, 2)
    unfoldM = tl.unfold(matrix, 2)

    # Slice out missing RNAseq positions
    selIDX = np.all(np.isfinite(unfoldM), axis=0)

    if cost:
        cost = np.sum(np.linalg.lstsq(kr, unfold.T, rcond=None)[1])
        cost += np.sum(np.linalg.lstsq(kr[selIDX, :], unfoldM[:, selIDX].T, rcond=None)[1])
        return cost

    C = np.linalg.lstsq(kr, unfold.T, rcond=None)[0].T
    tFac = tl.cp_tensor.CPTensor((None, [A, B, C]))
    D = np.linalg.lstsq(kr[selIDX, :], unfoldM[:, selIDX].T, rcond=None)[0].T
    tFac.mFactor = tl.cp_tensor.CPTensor((None, [A, B, D]))
    return tFac


def cost(pIn, tOrig, mOrig, r):
    return buildTensors(pIn, tOrig, mOrig, r, cost=True)


def perform_CMTF(tOrig: np.ndarray, mOrig: np.ndarray, r=5):
    """ Perform CMTF decomposition by direct optimization. """
    # Checks
    tOrig = np.array(tOrig, dtype=float, order="C")
    mOrig = np.array(mOrig, dtype=float, order="C")
    assert tOrig.ndim == 3
    assert mOrig.ndim == 3
    assert tOrig.shape[0] == mOrig.shape[0]
    assert tOrig.shape[1] == mOrig.shape[1]

    tFac = tl.cp_tensor.CPTensor((None, [np.random.rand(tOrig.shape[i], r) for i in range(tOrig.ndim)]))
    x0 = cp_to_vec(tFac)
    res = minimize(lambda x: cost(x, tOrig, mOrig, r), x0, method="L-BFGS-B", options={"maxiter": 8000})

    tFac = buildTensors(res.x, tOrig, mOrig, r)

    tFac.mFactor.normalize()
    mFactor = tl.cp_tensor.cp_flip_sign(tFac.mFactor, mode=2)
    tFac.normalize()
    tFac = tl.cp_tensor.cp_flip_sign(tFac, mode=2)
    tFac.mFactor = mFactor

    tFac.R2X = calcR2X(tFac, tOrig, mOrig)
    print(tFac.R2X)

    return tFac
