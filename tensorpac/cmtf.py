"""
Coupled Matrix Tensor Factorization
"""

import numpy as np
import tensorly as tl
from tensorly.tenalg import khatri_rao
from copy import deepcopy
from tensorly.decomposition._cp import initialize_cp, parafac
from fancyimpute import SoftImpute
from scipy.optimize import minimize
from .cp import cp_normalize


tl.set_backend('numpy')


def buildMat(tFac):
    """ Build the matrix in CMTF from the factors. """
    return tFac.factors[0] @ tFac.mFactor.T


def calcR2X(tFac, tIn=None, mIn=None):
    """ Calculate R2X. Optionally it can be calculated for only the tensor or matrix. """
    assert (tIn is not None) or (mIn is not None)

    vTop, vBottom = 0.0, 0.0

    if tIn is not None:
        tMask = np.isfinite(tIn)
        tIn = np.nan_to_num(tIn)
        vTop += np.linalg.norm(tl.cp_to_tensor(tFac) * tMask - tIn)**2.0
        vBottom += np.linalg.norm(tIn)**2.0
    if mIn is not None:
        mMask = np.isfinite(mIn)
        recon = tFac if isinstance(tFac, np.ndarray) else buildMat(tFac)
        mIn = np.nan_to_num(mIn)
        vTop += np.linalg.norm(recon * mMask - mIn)**2.0
        vBottom += np.linalg.norm(mIn)**2.0

    return 1.0 - vTop / vBottom


def tensor_degFreedom(tFac) -> int:
    """ Calculate the degrees of freedom within a tensor factorization. """
    deg = np.sum([f.size for f in tFac.factors])

    if hasattr(tFac, 'mFactor'):
        deg += tFac.mFactor.size

    return deg


def reorient_factors(tFac):
    """ This function ensures that factors are negative on at most one direction. """
    # Flip the types to be positive
    tMeans = np.sign(np.mean(tFac.factors[2], axis=0))
    tFac.factors[1] *= tMeans[np.newaxis, :]
    tFac.factors[2] *= tMeans[np.newaxis, :]

    # Flip the cytokines to be positive
    rMeans = np.sign(np.mean(tFac.factors[1], axis=0))
    tFac.factors[0] *= rMeans[np.newaxis, :]
    tFac.factors[1] *= rMeans[np.newaxis, :]

    if hasattr(tFac, 'mFactor'):
        tFac.mFactor *= rMeans[np.newaxis, :]
    return tFac


def sort_factors(tFac):
    """ Sort the components from the largest variance to the smallest. """
    tensor = deepcopy(tFac)

    # Variance separated by component
    norm = np.copy(tFac.weights)
    for factor in tFac.factors:
        norm *= np.sum(np.square(factor), axis=0)

    # Add the variance of the matrix
    if hasattr(tFac, 'mFactor'):
        norm += np.sum(np.square(tFac.factors[0]), axis=0) * np.sum(np.square(tFac.mFactor), axis=0)

    order = np.flip(np.argsort(norm))
    tensor.weights = tensor.weights[order]
    tensor.factors = [fac[:, order] for fac in tensor.factors]
    np.testing.assert_allclose(tl.cp_to_tensor(tFac), tl.cp_to_tensor(tensor), atol=1e-9)

    if hasattr(tFac, 'mFactor'):
        tensor.mFactor = tensor.mFactor[:, order]
        np.testing.assert_allclose(buildMat(tFac), buildMat(tensor), atol=1e-9)

    return tensor


def delete_component(tFac, compNum):
    """ Delete the indicated component. """
    tensor = deepcopy(tFac)
    compNum = np.array(compNum, dtype=int)

    # Assert that component # don't exceed range, and are unique
    assert np.amax(compNum) < tensor.rank
    assert np.unique(compNum).size == compNum.size

    tensor.rank -= compNum.size
    tensor.weights = np.delete(tensor.weights, compNum)

    if hasattr(tFac, 'mFactor'):
        tensor.mFactor = np.delete(tensor.mFactor, compNum, axis=1)

    tensor.factors = [np.delete(fac, compNum, axis=1) for fac in tensor.factors]
    return tensor


def censored_lstsq(A: np.ndarray, B: np.ndarray, uniqueInfo) -> np.ndarray:
    """Solves least squares problem subject to missing data.
    Note: uses a for loop over the missing patterns of B, leading to a
    slower but more numerically stable algorithm
    Args
    ----
    A (ndarray) : m x r matrix
    B (ndarray) : m x n matrix
    Returns
    -------
    X (ndarray) : r x n matrix that minimizes norm(M*(AX - B))
    """
    X = np.empty((A.shape[1], B.shape[1]))
    # Missingness patterns
    unique, uIDX = uniqueInfo

    for i in range(unique.shape[1]):
        uI = uIDX == i
        uu = np.squeeze(unique[:, i])

        Bx = B[uu, :]
        X[:, uI] = np.linalg.lstsq(A[uu, :], Bx[:, uI], rcond=-1)[0]
    return X.T


def initialize_cmtf(tensor: np.ndarray, matrix: np.ndarray, rank: int):
    r"""Initialize factors used in `parafac`.
    Parameters
    ----------
    tensor : ndarray
    rank : int
    Returns
    -------
    factors : CPTensor
        An initial cp tensor.
    """
    factors = [np.ones((tensor.shape[i], rank)) for i in range(tensor.ndim)]

    # SVD init mode 0
    unfold = tl.unfold(tensor, 0)
    unfold = np.hstack((unfold, matrix))

    if np.sum(~np.isfinite(unfold)) > 0:
        si = SoftImpute(max_rank=rank)
        unfold = si.fit_transform(unfold)

    factors[0] = np.linalg.svd(unfold)[0][:, :rank]

    unfold = tl.unfold(tensor, 1)
    unfold = unfold[:, np.all(np.isfinite(unfold), axis=0)]
    factors[1] = np.linalg.svd(unfold)[0]
    factors[1] = factors[1].take(range(rank), axis=1, mode="wrap")
    return tl.cp_tensor.CPTensor((None, factors))




def perform_CMTF(tOrig, mOrig=None, r=9, tol=1e-6):
    """ Perform CMTF decomposition. """
    assert tOrig.dtype == float
    if mOrig is not None:
        assert mOrig.dtype == float
    tFac = initialize_cmtf(tOrig, mOrig, r)

    # Pre-unfold
    unfolded = np.hstack((tl.unfold(tOrig, 0), mOrig))
    missingM = np.all(np.isfinite(mOrig), axis=1)
    R2X = -np.inf

    # Precalculate the missingness patterns
    uniqueInfo = np.unique(np.isfinite(unfolded.T), axis=1, return_inverse=True)

    for _ in range(40):
        tensor = np.nan_to_num(tOrig) + tl.cp_to_tensor(tFac) * np.isnan(tOrig)
        tFac = parafac(tensor, r, 200, init=tFac, verbose=False, fixed_modes=[0], mask=np.isfinite(tOrig))

        # Solve for the glycan matrix fit
        tFac.mFactor = np.linalg.lstsq(tFac.factors[0][missingM, :], mOrig[missingM, :], rcond=-1)[0].T

        # Solve for subjects factors
        kr = khatri_rao(tFac.factors, skip_matrix=0)
        kr = np.vstack((kr, tFac.mFactor))
        tFac.factors[0] = censored_lstsq(kr, unfolded.T, uniqueInfo)

        R2X_last = R2X
        R2X = calcR2X(tFac, tOrig, mOrig)
        assert R2X > 0.0

        if R2X - R2X_last < tol:
            break

    tFac = cp_normalize(tFac)
    tFac = reorient_factors(tFac)
    tFac = sort_factors(tFac)
    tFac.R2X = R2X

    print("R2X: " + str(tFac.R2X))

    return tFac



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

def perform_CMTF_direct(tOrig: np.ndarray, mOrig: np.ndarray, r=5):
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