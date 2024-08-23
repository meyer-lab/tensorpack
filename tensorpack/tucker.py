""" Tucker decomposition """

import numpy as np
from tensorly.decomposition import tucker


def tucker_decomp(tensor, num_comps: int):
    """ Performs Tucker decomposition.

    Parameters
    ----------
    tensor : xarray or ndarray
        multi-dimensional data input
    num_comps : int
        the number of components.

    Returns
    -------
    factors : list of lists
        containing tucker factorization object of each rank.
    min_err : list
        list of minimum errors of tensor reconstruction for each sum of components.
    min_err_rank : list of tuples
        list of the corresponding ranks combinations for the minimum error.
    """

    # if tensor is xarray...
    if type(tensor) is not np.ndarray:
        tensor = tensor.to_numpy()

    mask = np.isfinite(tensor)
    tensor_filled = np.nan_to_num(tensor)

    # step 1 with 1 component along every dimension
    start = [1] * tensor.ndim
    ff, errors = tucker(tensor_filled, rank=start, svd='randomized_svd', tol=1e-8, mask=mask, return_errors=True)
    factors = [ff]
    min_err = [errors[-1] ** 2.0]
    min_rank = [start]
    ranks = min_rank * tensor.ndim

    for _ in range(tensor.ndim * num_comps):
        fac = []
        err = []
        rnk = []
        for indx, val in enumerate(ranks):
            temp_rank = val.copy()
            temp_rank[indx] = val[indx] + 1

            if temp_rank[indx] > tensor.shape[indx]:
                continue

            # calculate error for this rank combination
            ff, errors = tucker(tensor_filled, rank=temp_rank, svd='randomized_svd', tol=1e-8, mask=mask, return_errors=True)
            fac.append(ff)
            err.append(errors[-1] ** 2.0)
            rnk.append(temp_rank)

        # pick the lowest error and continue with that
        min_err.append(min(err))
        min_rank.append(rnk[err.index(min(err))])
        factors.append(fac[err.index(min(err))])

        ranks = [min_rank[-1]] * tensor.ndim

    return factors, min_err, min_rank
