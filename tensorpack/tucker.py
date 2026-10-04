"""Tucker decomposition"""

import numpy as np
from tensorly.decomposition import tucker


def _fit_tucker(tensor_filled, mask, rank):
    """Fit one masked Tucker decomposition; return the fit and its squared error."""
    fit, errors = tucker(
        tensor_filled,
        rank=rank,
        svd="randomized_svd",
        tol=1e-8,
        mask=mask,
        return_errors=True,
    )
    return fit, errors[-1] ** 2.0


def tucker_decomp(tensor, num_comps: int):
    """Performs Tucker decomposition, greedily growing the rank of one mode at a time.

    Starting from rank 1 along every mode, each step increases the rank of
    whichever single mode lowers the reconstruction error the most. A mode is
    no longer considered once its rank reaches ``min(num_comps, size of that
    mode)``, so no mode is ever given a rank above ``num_comps`` and the search
    ends once every mode has reached its limit.

    Parameters
    ----------
    tensor : xarray or ndarray
        multi-dimensional data input
    num_comps : int
        the maximum rank to test along each mode.

    Returns
    -------
    factors : list of lists
        containing tucker factorization object of each rank.
    min_err : list
        list of minimum errors of tensor reconstruction for each rank combination.
    min_err_rank : list of lists
        list of the corresponding rank combinations (one rank per mode) for the
        minimum error. The first entry is all ones; each later entry differs
        from the previous one by +1 in exactly one mode.
    """

    # if tensor is xarray...
    if type(tensor) is not np.ndarray:
        tensor = tensor.to_numpy()

    mask = np.isfinite(tensor)
    tensor_filled = np.nan_to_num(tensor)
    max_rank = [min(num_comps, size) for size in tensor.shape]

    # step 1 with 1 component along every dimension
    rank = [1] * tensor.ndim
    fit, err = _fit_tucker(tensor_filled, mask, rank)
    factors = [fit]
    min_err = [err]
    min_rank = [rank]

    while True:
        candidates = []
        for mode in range(tensor.ndim):
            if rank[mode] >= max_rank[mode]:
                continue
            temp_rank = rank.copy()
            temp_rank[mode] += 1
            candidates.append((temp_rank, *_fit_tucker(tensor_filled, mask, temp_rank)))

        if not candidates:
            break

        # pick the lowest error and continue with that
        rank, fit, err = min(candidates, key=lambda c: c[2])
        factors.append(fit)
        min_err.append(err)
        min_rank.append(rank)

    return factors, min_err, min_rank
