""" Tucker decomposition """

from tensorly.decomposition import tucker
import numpy as np
import tensorly as tl
from .linalg import calcR2X_TnB

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
    factors = [tucker(tensor_filled, rank=start, svd='randomized_svd', mask=mask)]
    err_top, err_bot = calcR2X_TnB(tensor, tl.tucker_to_tensor(factors[0]))

    min_err = [1.0 - err_top / err_bot]
    min_rank = [start]
    ranks = min_rank * tensor.ndim

    for _ in range(tensor.ndim * num_comps):

        fac = []
        err = []
        rnk = []
        for indx, val in enumerate(ranks):
            temp_rank = val.copy()
            temp_rank[indx] = val[indx] + 1

            # calculate error for this rank combination
            fac.append(tucker(tensor_filled, rank=temp_rank, svd='randomized_svd', mask=mask))
            err_top, err_bot = calcR2X_TnB(tensor, tl.tucker_to_tensor(factors[0]))

            err.append(1.0 - err_top / err_bot)
            rnk.append(temp_rank)

        # pick the lowest error and continue with that
        min_err.append(min(err))
        min_rank.append(rnk[err.index(min(err))])
        factors.append(fac[err.index(min(err))])

        ranks = [min_rank[-1]] * tensor.ndim

    return factors, min_err, min_rank
