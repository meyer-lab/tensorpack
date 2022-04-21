""" Tucker decomposition """

from tensorly.decomposition import tucker
import numpy as np
import itertools as it
import tensorly as tl

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


    # step 1 with 1 component along every dimension
    start = [1] * tensor.ndim
    factors = [tucker(tensor, rank=start, svd='randomized_svd')]
    min_err = [(tl.norm(tl.tucker_to_tensor(factors[0]) - tensor) ** 2) / tl.norm(tensor) ** 2]
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
            fac.append(tucker(tensor, rank=temp_rank, svd='randomized_svd'))
            err.append((tl.norm(tl.tucker_to_tensor(fac[-1]) - tensor) ** 2) / tl.norm(tensor) ** 2)
            rnk.append(temp_rank)

        # pick the lowest error and continue with that
        min_err.append(min(err))
        min_rank.append(rnk[err.index(min(err))])
        factors.append(fac[err.index(min(err))])

        ranks = [min_rank[-1]] * tensor.ndim
        if sum(ranks[-1]) >= tensor.ndim * num_comps:
            break

    # check if specified ranks are already included in the calculations
    specified_ranks = [[i] * tensor.ndim for i in range(1, num_comps+1)]
    for rnks in specified_ranks:
        if rnks not in min_rank:
            factors.append(tucker(tensor, rank=rnks, svd='randomized_svd'))
            min_err.append((tl.norm(tl.tucker_to_tensor(fac[-1]) - tensor) ** 2) / tl.norm(tensor) ** 2)
            min_rank.append(rnks)

    return factors, min_err, min_rank
