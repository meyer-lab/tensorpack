""" Tucker decomposition """

from tensorly.decomposition import tucker
import numpy as np
import itertools as it
import tensorly as tl

def tucker_rank(least_comps: int, most_comps: int):
    """ Create list of tuples to pass to tucker as the rank. Depending on the number of dimensions of the tensor.
    :param least_comps: least number of total components = the number of dimensions of the tensor.
    :param most_comps: most number of total components = highest_rank - 1.
    :return ranks: list of lists of permutations of ranks for each sum of components.
    Example:
    least_comps = 3, most_comps = 5
    -> return [ [(1, 1, 1)], 
                [(1, 1, 2), (1, 2, 1), (2, 1, 1)] ]
    """

    ranks = []
    for i in range(least_comps, most_comps):
        # combinations
        choices = [pair for pair in it.combinations_with_replacement(range(1, most_comps-1), least_comps) if sum(pair) == i]
        permutation = []
        # permutations of the choices, remove duplicates
        for sets in choices:
            permutation.append(list(set([pars for pars in it.permutations(sets)])))
        ranks.append(list(it.chain(*permutation)))

    return ranks

def perform_tucker(tensor, num_comps: int):
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

    ranks = tucker_rank(tensor.ndim, num_comps+1)

    # if tensor is xarray...
    if type(tensor) is not np.ndarray:
        tensor = tensor.to_numpy()

    min_err = []
    min_err_rank = []
    factors = []

    for total_rank in ranks:
        error = []
        facs = []

        for eachCP_rank in total_rank:
            facs.append(tucker(tensor, rank=eachCP_rank, svd='randomized_svd'))

            # calculate error for this rank combination
            error.append((tl.norm(tl.tucker_to_tensor(facs[-1]) - tensor) ** 2) / tl.norm(tensor) ** 2)

        # append the least error of total_rank
        min_err.append(min(error))

        # append the rank combination for all dimensions for the min error
        min_err_rank.append(total_rank[error.index(min(error))])

        # append the tucker decomposition object for the minimum error for a specific combination.
        factors.append(facs[error.index(min(error))])

    return factors, min_err, min_err_rank
