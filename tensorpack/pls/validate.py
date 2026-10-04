import numpy as np
from sklearn.model_selection import LeaveOneOut

from tensorpack.pls.tpls import tPLS


def get_q2y(pls_tensor):
    """
    Calculates Q2Y for a fitted PLS tensor.

    Args:
        pls_tensor (tPLS): fitted PLS tensor.

    Returns:
        Q2Y (float): Q2Y of PLS tensor applied to fitted dataset.
    """
    assert (
        pls_tensor.original_X is not None
    ), "PLS Tensor must be fit prior to calculating Q2Y"
    X = pls_tensor.original_X
    Y = pls_tensor.original_Y
    q2y_plsr = tPLS(pls_tensor.n_components)

    loo = LeaveOneOut()
    Y_pred = np.zeros(Y.shape)
    Y_actual = np.zeros(Y.shape)
    for train_index, test_index in loo.split(X, Y):
        X_train, Y_train = X[train_index], Y[train_index]
        X_test, Y_test = X[test_index], Y[test_index]
        q2y_plsr.fit(X_train, Y_train)

        Y_pred[test_index] = q2y_plsr.predict(X_test)
        Y_actual[test_index] = Y_test

    numerator = (Y_pred - Y_actual) ** 2
    denominator = (Y_actual) ** 2
    return 1 - numerator.sum() / denominator.sum()
