import numpy as np


def mlstsq(A: np.ndarray, B: np.ndarray, uniqueInfo=None) -> np.ndarray:
    """ Solve min[Ax - b]_2 while checking missing values
        Args
        ----
        A (ndarray) : m x r matrix
        B (ndarray) : m x n matrix
        Returns
        -------
        X (ndarray) : r x n matrix that minimizes norm(M*(AX - B))
    """
    # contain missing values
    if np.any(np.isnan(B)):
        """ Solves least squares problem subject to missing data.
            Note: uses a for loop over the missing patterns of B, leading to a
            slower but more numerically stable algorithm """
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
        return X
    # does not contain missing values
    else:
        return np.linalg.lstsq(A, B, rcond=-1)[0]

