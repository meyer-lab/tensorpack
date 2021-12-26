import tensorly as tl
import numpy as np
from tensorly.random import random_cp
from tensorpack.cmtf import perform_CP, calcR2X


def testcase(shape: tuple, rankOrig: int, rankNew: int):

    '''
    Creates a random CP tensor and converts it into a full tensor.
    Then, it tests the CP-decomposition method to see if the R2X results correctly.
    '''

    tFacOrig = random_cp(shape, rankOrig, full=False)
    tOrig = tl.cp_to_tensor(tFacOrig)

    originalR2X = calcR2X(tFacOrig, tOrig)
    assert originalR2X == 1

    tFacNew = perform_CP(tOrig, r=rankNew)
    newR2X = calcR2X(tFacNew, tOrig)

    return newR2X
