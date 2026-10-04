import numpy as np
import tensorly as tl


def make_synthetic_test(
    cp_tensor: tl.cp_tensor, test_samples: int, error: float = 0, seed: int = 215
):
    """
    Generates test set from given factors.

    Parameters:
        cp_tensor (tl.cp_tensor): CP tensor
        test_samples (int): samples in testing set
        error (float, default: 0): standard error of added gaussian noise
        seed (int, default: 215): seed for random number generator

    Returns:
        x_test (np.array): tensor of measurements
        y_test (np.array): response variables
        test_tensor (tl.cp_tensor): CP tensor of x_test, y_test
    """
    rng = np.random.default_rng(seed)

    test_factors = cp_tensor.factors
    test_factors[0] = rng.normal(0, 1, size=(test_samples, cp_tensor.rank))
    test_tensor = tl.cp_tensor.CPTensor((None, test_factors))
    test_tensor.y_factor = cp_tensor.y_factor

    x_test = tl.cp_to_tensor(test_tensor)
    x_test += rng.normal(0, error, size=test_tensor.shape)
    y_test = tl.dot(test_tensor.factors[0], cp_tensor.y_factor.T)
    y_test += rng.normal(0, error, size=y_test.shape)

    return x_test, y_test, test_tensor


def import_synthetic(
    train_dimensions: tuple,
    n_response: int,
    n_latent: int,
    error: float = 0,
    seed: int = 215,
):
    """
    Generates synthetic data.

    Parameters:
        train_dimensions (tuple): dimensions of training data
        n_response (int): number of response variables
        n_latent (int): number of latent variables in synthetic data
        error (float, default: 0): standard error of added gaussian noise
        seed (int, default: 215): seed for random number generator

    Returns:
        x (np.array): tensor of measurements
        y (np.array): response variables
        cp_tensor (tl.cp_tensor): CP tensor of x, y
    """
    rng = np.random.default_rng(seed)

    x_factors = [rng.normal(0, 1, size=(train_dimensions[0], n_latent))]
    y_factor = rng.normal(0, 1, size=(n_response, n_latent))

    for dimension in train_dimensions[1:]:
        x_factors.append(rng.normal(0, 1, size=(dimension, n_latent)))

    cp_tensor = tl.cp_tensor.CPTensor((None, x_factors))
    cp_tensor.y_factor = y_factor

    x = tl.cp_to_tensor(cp_tensor)
    x += rng.normal(0, error, size=train_dimensions)

    y = tl.dot(cp_tensor.factors[0], cp_tensor.y_factor.T)
    y += rng.normal(0, error, size=(train_dimensions[0], n_response))

    if y.shape[1] == 1:
        y = y.flatten()

    return x, y, cp_tensor
