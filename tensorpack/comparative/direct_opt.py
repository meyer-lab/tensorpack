"""
Tensor decomposition methods
"""
import numpy as np
import jax.numpy as jnp
from jax import jit, grad, jvp
from jax.config import config
from scipy.optimize import minimize
import tensorly as tl
from tensorly.decomposition import parafac
from tensorly.kruskal_tensor import KruskalTensor, kruskal_normalise
from .dataImport import createCube

tl.set_backend('numpy')
config.update("jax_enable_x64", True)

def calcR2X(tensorIn, matrixIn, tensorFac, matrixFac):
    """ Calculate R2X. """
    tErr = np.nanvar(tl.kruskal_to_tensor(tensorFac) - tensorIn)
    mErr = np.nanvar(tl.kruskal_to_tensor(matrixFac) - matrixIn)
    return 1.0 - (tErr + mErr) / (np.nanvar(tensorIn) + np.nanvar(matrixIn))


def reorient_factors(tensorFac, matrixFac):
    """ This function ensures that factors are negative on at most one direction. """
    for jj in range(1, len(tensorFac)):
        # Calculate the sign of the current factor in each component
        means = np.sign(np.mean(tensorFac[jj], axis=0))

        # Update both the current and last factor
        tensorFac[0] *= means[np.newaxis, :]
        matrixFac[0] *= means[np.newaxis, :]
        matrixFac[1] *= means[np.newaxis, :]
        tensorFac[jj] *= means[np.newaxis, :]
    return tensorFac, matrixFac


def buildTensors(pIn, tensor, matrix, tmask, r):
    """ Use parameter vector to build kruskal tensors. """
    assert tensor.shape[0] == matrix.shape[0]
    nn = np.cumsum(tensor.shape) * r
    A = jnp.reshape(pIn[:nn[0]], (tensor.shape[0], r))
    B = jnp.reshape(pIn[nn[0]:nn[1]], (tensor.shape[1], r))
    C = jnp.reshape(pIn[nn[1]:], (tensor.shape[2], r))

    # Solve for the glycan matrix fit
    selPat = np.all(np.isfinite(matrix), axis=1)
    G = jnp.linalg.lstsq(A[selPat, :], matrix[selPat, :])[0]

    return KruskalTensor((None, [A, B, C])), KruskalTensor((None, [A, G.T]))


def cost(pIn, tensor, matrix, tmask, r):
    tl.set_backend('jax')
    tensF, matF = buildTensors(pIn, tensor, matrix, tmask, r)
    matrix = matrix.copy()
    mmask = np.isnan(matrix)
    matrix[mmask] = 0.0
    cost = jnp.linalg.norm(tl.kruskal_to_tensor(tensF, mask=1 - tmask) - tensor) # Tensor cost
    cost += jnp.linalg.norm(tl.kruskal_to_tensor(matF, mask=1 - mmask) - matrix) # Matrix cost
    cost += 1e-9 * jnp.linalg.norm(pIn)
    tl.set_backend('numpy')
    return cost


def perform_CMTF(tensorOrig=None, matrixOrig=None, r=6):
    """ Perform CMTF decomposition. """
    if tensorOrig is None:
        tensorOrig, matrixIn = createCube()

    tensorIn = tensorOrig.copy()
    tmask = np.isnan(tensorIn)
    tensorIn[tmask] = 0.0

    cost_jax = jit(cost, static_argnums=(1, 2, 3, 4))
    cost_grad = jit(grad(cost, 0), static_argnums=(1, 2, 3, 4))

    def costt(*args):
        return np.array(cost_jax(*args))

    def gradd(*args):
        return np.array(cost_grad(*args))

    CPinit = parafac(tensorIn.copy(), r, mask=tmask, n_iter_max=50, orthogonalise=10)
    x0 = np.concatenate((np.ravel(CPinit.factors[0]), np.ravel(CPinit.factors[1]), np.ravel(CPinit.factors[2])))

    rgs = (tensorIn, matrixIn, tmask, r)
    res = minimize(costt, x0, method='L-BFGS-B', jac=gradd, args=rgs, options={"maxiter": 50000})
    tensorFac, matrixFac = buildTensors(res.x, tensorIn, matrixIn, tmask, r)
    tensorFac = kruskal_normalise(tensorFac)
    matrixFac = kruskal_normalise(matrixFac)

    # Reorient the later tensor factors
    tensorFac.factors, matrixFac.factors = reorient_factors(tensorFac.factors, matrixFac.factors)

    R2X = calcR2X(tensorOrig, matrixIn, tensorFac, matrixFac)

    for ii in range(3):
        tensorFac.factors[ii] = np.array(tensorFac.factors[ii])
    for ii in range(2):
        matrixFac.factors[ii] = np.array(matrixFac.factors[ii])

    return tensorFac, matrixFac, R2X