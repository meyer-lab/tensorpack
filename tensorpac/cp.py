"""
Canonical Polyadic
"""

import numpy as np
import tensorly as tl
from tensorly.tenalg import khatri_rao
from copy import deepcopy
from tensorly.decomposition._cp import initialize_cp, parafac
from fancyimpute import SoftImpute
from .cmtf import calcR2X, censored_lstsq, reorient_factors, sort_factors


def cp_normalize(tFac):
    """ Normalize the factors using the inf norm. """
    for i, factor in enumerate(tFac.factors):
        scales = np.linalg.norm(factor, ord=np.inf, axis=0)
        tFac.weights *= scales
        if i == 0 and hasattr(tFac, 'mFactor'):
            tFac.mFactor *= scales

        tFac.factors[i] /= scales

    return tFac

def initialize_cp(tensor: np.ndarray, rank: int):
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
    contain_missing = (np.sum(~np.isfinite(tensor)) > 0)

    # SVD init mode whose size is larger than rank
    for mode in range(tensor.ndim):
        if tensor.shape[mode] >= rank:
            unfold = tl.unfold(tensor, mode)
            if contain_missing:
                si = SoftImpute(max_rank=rank)
                unfold = si.fit_transform(unfold)

            factors[mode] = np.linalg.svd(unfold)[0][:, :rank]

    return tl.cp_tensor.CPTensor((None, factors))


def perform_CP(tOrig, r=6, tol=1e-6):
    """ Perform CP decomposition. """
    tFac = initialize_cp(tOrig, r)

    # Pre-unfold
    unfolded = [tl.unfold(tOrig, i) for i in range(tOrig.ndim)]

    R2X_last = -np.inf
    tFac.R2X = calcR2X(tFac, tOrig)

    # Precalculate the missingness patterns
    uniqueInfo = [np.unique(np.isfinite(B.T), axis=1, return_inverse=True) for B in unfolded]

    for ii in range(2000):
        # Solve on each mode
        for m in range(len(tFac.factors)):
            kr = khatri_rao(tFac.factors, skip_matrix=m)
            tFac.factors[m] = censored_lstsq(kr, unfolded[m].T, uniqueInfo[m])

        if ii % 2 == 0:
            R2X_last = tFac.R2X
            tFac.R2X = calcR2X(tFac, tOrig)
            assert tFac.R2X > 0.0

        if tFac.R2X - R2X_last < tol:
            break

    tFac = cp_normalize(tFac)
    tFac = reorient_factors(tFac)

    if r > 1:
        tFac = sort_factors(tFac)
    print(tFac.R2X)

    return tFac