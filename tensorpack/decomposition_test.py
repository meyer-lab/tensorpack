import tensorly as tl
import numpy as np
from tensorly.random import random_cp
from .cmtf import perform_CP, calcR2X


def test_factorization(shape: tuple, rankOrig: int, rankNew: int):

    tFacOrig = random_cp(shape, rankOrig, full=False)
    tOrig = tl.cp_to_tensor(tFacOrig)

    originalR2X = calcR2X(tFacOrig, tOrig)
    assert originalR2X == 1

    tFacNew = perform_CP(tOrig, r=rankNew)
    newR2X = calcR2X(tFacNew, tOrig)

    return newR2X
