import numpy as np

from tensorpack.pls.synthetic import import_synthetic, make_synthetic_test

TENSOR_DIMENSIONS = (100, 38, 65)
N_RESPONSE = 4
N_LATENT = 8


def test_synthetic_dimensions():
    x, y, cp_tensor = import_synthetic(TENSOR_DIMENSIONS, N_RESPONSE, N_LATENT, error=0)

    assert all(factor.shape[1] == N_LATENT for factor in cp_tensor.factors)
    assert cp_tensor.y_factor.shape[1] == N_LATENT
    assert x.shape == TENSOR_DIMENSIONS
    assert y.shape == (TENSOR_DIMENSIONS[0], N_RESPONSE)


def test_synthetic_test_dimensions():
    n_test = 10
    _x, _y, cp_tensor = import_synthetic(
        TENSOR_DIMENSIONS, N_RESPONSE, N_LATENT, error=0
    )
    _x_test, _y_test, test_tensor = make_synthetic_test(cp_tensor, n_test, 0)

    assert cp_tensor.factors[0].shape[1] == test_tensor.factors[0].shape[1]
    assert test_tensor.factors[0].shape[0] == n_test


def test_reproducibility():
    x1, y1, _cp_tensor1 = import_synthetic(
        TENSOR_DIMENSIONS, N_RESPONSE, N_LATENT, error=0, seed=42
    )
    x2, y2, _cp_tensor2 = import_synthetic(
        TENSOR_DIMENSIONS, N_RESPONSE, N_LATENT, error=0, seed=42
    )
    x3, y3, _cp_tensor3 = import_synthetic(
        TENSOR_DIMENSIONS, N_RESPONSE, N_LATENT, error=0, seed=43
    )

    assert np.array_equal(x1, x2)
    assert np.array_equal(y1, y2)
    assert not np.array_equal(x1, x3)
    assert not np.array_equal(y1, y3)


def test_shared_factor():
    x, y, cp_tensor = import_synthetic((10, 10), 10, 10, error=0, seed=42)

    inv_x_factor = np.linalg.inv(cp_tensor.factors[1].T)
    inv_y_factor = np.linalg.inv(cp_tensor.y_factor.T)
    assert np.allclose(np.matmul(x, inv_x_factor), np.matmul(y, inv_y_factor))
