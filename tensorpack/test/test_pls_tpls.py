import pytest
import numpy as np
from numpy.testing import assert_allclose
from sklearn.decomposition import PCA
import tensorly as tl
from tensorly.cp_tensor import CPTensor, cp_normalize
from tensorly.metrics.factors import congruence_coefficient

from tensorpack.pls.synthetic import import_synthetic
from tensorpack.pls.tpls import tPLS


TENSOR_DIMENSIONS = (100, 38, 65)
N_RESPONSE = 4
N_LATENT = 8


# Supporting Functions


def _get_standard_synthetic():
    x, y, cp_tensor = import_synthetic(TENSOR_DIMENSIONS, N_RESPONSE, N_LATENT)
    pls = tPLS(N_LATENT)
    pls.fit(x, y)
    return x, y, cp_tensor, pls


# Class Structure Tests


def test_factor_normality():
    x, y, _, pls = _get_standard_synthetic()
    for x_factor in pls.X_factors[1:]:
        assert_allclose(tl.norm(x_factor, axis=0), 1)
    for y_factor in pls.Y_factors[1:]:
        assert_allclose(tl.norm(y_factor, axis=0), 1)


# This method should test for factor hyper-orthogonality; components seem
# very loosely hyper-orthogonal (cut-off of 1E-2 is generous).
def test_factor_orthogonality():
    x, y, _, pls = _get_standard_synthetic()
    x_cp = CPTensor((None, pls.X_factors))
    x_cp = cp_normalize(x_cp)

    for component_1 in range(x_cp.rank):
        for component_2 in range(component_1 + 1, x_cp.rank):
            factor_product = 1
            for factor in x_cp.factors:
                factor_product *= np.dot(factor[:, component_1], factor[:, component_2])
            assert abs(factor_product) < 1e-2


def test_consistent_components():
    x, y, _, pls = _get_standard_synthetic()

    for x_factor in pls.X_factors:
        assert x_factor.shape[1] == N_LATENT

    for y_factor in pls.Y_factors:
        assert y_factor.shape[1] == N_LATENT


# Dimension Compatibility Tests


@pytest.mark.parametrize("idims", [(2, 1), (3, 1), (4, 1), (2, 4), (3, 4), (4, 4)])
def _test_dimension_compatibility(idims):
    x_rank, n_response = idims
    x, y, _ = import_synthetic(tuple([100] * x_rank), n_response, N_LATENT)
    try:
        pls = tPLS(N_LATENT)
        pls.fit(x, y)
    except ValueError:
        raise AssertionError(
            f"Fit failed for {len(x.shape)}-dimensional tensor with "
            f"{n_response} response variables in y"
        )


# Decomposition Accuracy Tests


def test_same_x_y():
    x, _, _ = import_synthetic((100, 100), N_RESPONSE, N_LATENT)
    pls = tPLS(N_LATENT)
    pca = PCA(N_LATENT)

    pls.fit(x, x)
    scores = pca.fit_transform(x)

    assert_allclose(pls.X_factors[0], pls.Y_factors[0], rtol=0, atol=1e-4)
    assert_allclose(pls.X_factors[1], pls.Y_factors[1], rtol=0, atol=1e-4)
    assert congruence_coefficient(pls.X_factors[0], scores)[0] > 0.95
    assert congruence_coefficient(pls.X_factors[1], pca.components_.T)[0] > 0.95


def test_zero_covariance_x():
    x, y, _ = import_synthetic(TENSOR_DIMENSIONS, N_RESPONSE, N_LATENT)
    x[:, 0, :] = 1
    pls = tPLS(N_LATENT)
    pls.fit(x, y)

    assert_allclose(pls.X_factors[1][0, :], 0)


@pytest.mark.parametrize("idims", [(3, 1), (4, 1), (3, 4), (4, 2)])
def _test_decomposition_accuracy(idims):
    x_rank, n_response = idims
    x, y, true_cp = import_synthetic(tuple([100] * x_rank), n_response, N_LATENT)
    pls = tPLS(N_LATENT)
    pls.fit(x, y)

    for pls_factor, true_factor in zip(pls.X_factors, true_cp.factors):
        assert congruence_coefficient(pls_factor, true_factor)[0] > 0.95

    assert congruence_coefficient(pls.Y_factors[1], true_cp.y_factor)[0] > 0.95


def _test_increasing_R2X(X, Y, info=""):
    tpls = tPLS(12)
    tpls.fit(X, Y)
    assert np.all(np.diff(tpls.R2X) >= 0.0), "R2X is not monotonically increasing"
    assert np.all(np.diff(tpls.R2Y) >= 0.0), (
        f"R2Y is not monotonically increasing. "
        f"Streak till {np.where(np.diff(tpls.R2Y) <= 0.0)[0][0] + 1}-th component, "
        f"R2Y = {tpls.R2Y[np.where(np.diff(tpls.R2Y) <= 0.0)[0][0]]}. "
        f"Y shape = {Y.shape}. {info}"
    )


@pytest.mark.parametrize("n_response", [5, 7, 9])
def test_increasing_R2X_random(n_response):
    X = np.random.rand(20, 8, 6, 4)
    Y = np.random.rand(20, n_response)
    _test_increasing_R2X(X, Y)


@pytest.mark.parametrize("n_response", [5, 7, 9])
def test_increasing_R2X(n_response, n_latent=5):
    X, Y, _ = import_synthetic((20, 8, 6, 4), n_response, n_latent)
    _test_increasing_R2X(X, Y, info=f"n_latent = {n_latent}")


def test_transform():
    """Test transform the original X and Y will give the first factors"""
    X = np.random.rand(20, 8, 6, 4)
    Y = np.random.rand(20, 5)
    tpls = tPLS(6)
    tpls.fit(X, Y)
    rord = np.arange(20)
    np.random.shuffle(rord)
    X_scores, Y_scores = tpls.transform(X[rord, :], Y[rord, :])
    assert np.allclose(X_scores, tpls.X_factors[0][rord, :])
    assert np.allclose(Y_scores, tpls.Y_factors[0][rord, :])
