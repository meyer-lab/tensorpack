"""
Coupled Matrix Tensor Factorization
"""

import numpy as np
from tensorly import partial_svd
import tensorly as tl
from tensorly.tenalg import khatri_rao
from copy import deepcopy
from tensorly.decomposition._cp import initialize_cp
from tqdm import tqdm
from .SVD_impute import IterativeSVD


tl.set_backend('numpy')


def buildMat(tFac):
    """ Build the matrix in CMTF from the factors. """
    if hasattr(tFac, 'mWeights'):
        return tFac.factors[0] @ (tFac.mFactor * tFac.mWeights).T
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
        norm += np.sum(np.square(tFac.factors[0]), axis=0) * np.sum(np.square(tFac.mFactor), axis=0) * tFac.mWeights

    order = np.flip(np.argsort(norm))
    tensor.weights = tensor.weights[order]
    tensor.factors = [fac[:, order] for fac in tensor.factors]
    np.testing.assert_allclose(tl.cp_to_tensor(tFac), tl.cp_to_tensor(tensor), atol=1e-9)

    if hasattr(tFac, 'mFactor'):
        tensor.mFactor = tensor.mFactor[:, order]
        tensor.mWeights = tensor.mWeights[order]
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
        tensor.mWeights = np.delete(tensor.mWeights, compNum)

    tensor.factors = [np.delete(fac, compNum, axis=1) for fac in tensor.factors]
    return tensor


def censored_lstsq(A: np.ndarray, B: np.ndarray, uniqueInfo=None) -> np.ndarray:
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
    if uniqueInfo is None:
        unique, uIDX = np.unique(np.isfinite(B), axis=1, return_inverse=True)
    else:
        unique, uIDX = uniqueInfo

    for i in range(unique.shape[1]):
        uI = uIDX == i
        uu = np.squeeze(unique[:, i])

        Bx = B[uu, :]
        X[:, uI] = np.linalg.lstsq(A[uu, :], Bx[:, uI], rcond=-1)[0]
    return X.T


def cp_normalize(tFac):
    """ Normalize the factors using the inf norm. """
    for i, factor in enumerate(tFac.factors):
        scales = np.linalg.norm(factor, ord=np.inf, axis=0)
        tFac.weights *= scales
        if i == 0 and hasattr(tFac, 'mFactor'):
            mScales = np.linalg.norm(tFac.mFactor, ord=np.inf, axis=0)
            tFac.mWeights = scales * mScales
            tFac.mFactor /= mScales

        tFac.factors[i] /= scales

    return tFac


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
        si = IterativeSVD(rank=rank, random_state=1)
        unfold = si.fit_transform(unfold)
        factors[0] = si.U
    else:
        factors[0] = np.linalg.svd(unfold)[0][:, :rank]

    unfold = tl.unfold(tensor, 1)
    unfold = unfold[:, np.all(np.isfinite(unfold), axis=0)]
    factors[1] = np.linalg.svd(unfold)[0]
    factors[1] = factors[1].take(range(rank), axis=1, mode="wrap")
    return tl.cp_tensor.CPTensor((None, factors))


def initialize_cp(tensor: np.ndarray, rank: int):
    """Initialize factors used in `parafac`.
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
    contain_missing = (np.sum(~np.isfinite(tensor)) > 0)

    # SVD init mode whose size is larger than rank
    for mode in range(tensor.ndim):
        if tensor.shape[mode] >= rank:
            unfold = tl.unfold(tensor, mode)
            if contain_missing:
                si = IterativeSVD(rank)
                unfold = si.fit_transform(unfold)

            factors[mode] = partial_svd(unfold, rank, flip=True)[0]

    return tl.cp_tensor.CPTensor((None, factors))


def perform_CP(tOrig, r=6, tol=1e-6, maxiter=50, progress=False, callback=None):
    """ Perform CP decomposition. """
    if callback: callback.begin()
    tFac = initialize_cp(tOrig, r)

    # Pre-unfold
    unfolded = [tl.unfold(tOrig, i) for i in range(tOrig.ndim)]

    R2X_last = -np.inf
    tFac.R2X = calcR2X(tFac, tOrig)

    # Precalculate the missingness patterns
    uniqueInfo = [np.unique(np.isfinite(B.T), axis=1, return_inverse=True) for B in unfolded]

    tq = tqdm(range(maxiter), disable=(not progress))
    for i in tq:
        # Solve on each mode
        for m in range(len(tFac.factors)):
            kr = khatri_rao(tFac.factors, skip_matrix=m)
            tFac.factors[m] = censored_lstsq(kr, unfolded[m].T, uniqueInfo[m])

        R2X_last = tFac.R2X
        tFac.R2X = calcR2X(tFac, tOrig)
        tq.set_postfix(R2X=tFac.R2X, delta=tFac.R2X - R2X_last, refresh=False)
        assert tFac.R2X > 0.0
        if callback: callback(tFac)

        if tFac.R2X - R2X_last < tol:
            break

    tFac = cp_normalize(tFac)
    tFac = reorient_factors(tFac)

    if r > 1:
        tFac = sort_factors(tFac)
    
    return tFac


def perform_CMTF(tOrig, mOrig, r=9, tol=1e-6, maxiter=50, progress=True, callback=None):
    """ Perform CMTF decomposition. """
    assert tOrig.dtype == float
    assert mOrig.dtype == float
    if callback: callback.begin()
    tFac = initialize_cmtf(tOrig, mOrig, r)

    # Pre-unfold
    unfolded = np.hstack((tl.unfold(tOrig, 0), mOrig))
    missingM = np.all(np.isfinite(mOrig), axis=1)
    assert np.sum(missingM) >= 1, "mOrig must contain at least one complete row"
    R2X = -np.inf

    # Precalculate the missingness patterns
    uniqueInfo = np.unique(np.isfinite(unfolded.T), axis=1, return_inverse=True)

    tq = tqdm(range(maxiter), disable=(not progress))
    for _ in tq:
        for m in [1, 2]:
            kr = khatri_rao(tFac.factors, skip_matrix=m)
            tFac.factors[m] = censored_lstsq(kr, tl.unfold(tOrig, m).T)

        # Solve for the glycan matrix fit
        tFac.mFactor = np.linalg.lstsq(tFac.factors[0][missingM, :], mOrig[missingM, :], rcond=-1)[0].T

        # Solve for subjects factors
        kr = khatri_rao(tFac.factors, skip_matrix=0)
        kr = np.vstack((kr, tFac.mFactor))
        tFac.factors[0] = censored_lstsq(kr, unfolded.T, uniqueInfo)

        R2X_last = R2X
        R2X = calcR2X(tFac, tOrig, mOrig)
        tq.set_postfix(R2X=R2X, delta=R2X - R2X_last, refresh=False)
        assert R2X > 0.0
        if callback: callback(tFac)

        if R2X - R2X_last < tol:
            break

    assert not np.all(tFac.mFactor == 0.0)
    tFac = cp_normalize(tFac)
    tFac = reorient_factors(tFac)
    tFac = sort_factors(tFac)
    tFac.R2X = R2X

    return tFac