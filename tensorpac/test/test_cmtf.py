"""
Unit test file.
"""
import numpy as np
import tensorly as tl
from tensorly.cp_tensor import _validate_cp_tensor
from tensorly.random import random_cp
from ..cmtf import perform_CMTF, delete_component, calcR2X, buildMatrix, sort_factors

def createCube():
    return np.random.rand(10, 20, 25), np.random.rand(10, 15)


def test_R2X():
    """ Test to ensure R2X for higher components is larger. """
    arr = []
    tensor, matrix = createCube()
    for i in range(1, 5):
        facT = perform_CMTF(tensor, matrix, r=i)
        assert np.all(np.isfinite(facT.factors[0]))
        assert np.all(np.isfinite(facT.factors[1]))
        assert np.all(np.isfinite(facT.factors[2]))
        assert np.all(np.isfinite(facT.mFactor))
        arr.append(facT.R2X)
    for j in range(len(arr) - 1):
        assert arr[j] < arr[j + 1]
    # confirm R2X is >= 0 and <=1
    assert np.min(arr) >= 0
    assert np.max(arr) <= 1


def test_cp():
    """ Test that the CP decomposition code works. """
    tensor, _ = createCube()
    facT = perform_CMTF(tensor, r=6)


def test_delete():
    """ Test deleting a component results in a valid tensor. """
    tOrig, mOrig = createCube()
    facT = perform_CMTF(tOrig, mOrig, r=4)

    fullR2X = calcR2X(facT, tOrig, mOrig)

    for ii in range(facT.rank):
        facTdel = delete_component(facT, ii)
        _validate_cp_tensor(facTdel)

        delR2X = calcR2X(facTdel, tOrig, mOrig)

        assert delR2X < fullR2X


def test_sort():
    """ Test that sorting does not affect anything. """
    tOrig, mOrig = createCube()

    tFac = random_cp(tOrig.shape, 3)
    tFac.mFactor = np.random.randn(mOrig.shape[1], 3)

    R2X = calcR2X(tFac, tOrig, mOrig)
    tRec = tl.cp_to_tensor(tFac)
    mRec = buildMatrix(tFac)

    tFac = sort_factors(tFac)
    sR2X = calcR2X(tFac, tOrig, mOrig)
    stRec = tl.cp_to_tensor(tFac)
    smRec = buildMatrix(tFac)

    np.testing.assert_allclose(R2X, sR2X)
    np.testing.assert_allclose(tRec, stRec)
    np.testing.assert_allclose(mRec, smRec)
