"""
Testing Decomposition
"""

import numpy as np
import os
import tensorly as tl
from tensorly.random import random_cp
from ..decomposition import Decomposition
from .atyeo import createCube
from ..cmtf import perform_CP, calcR2X

def test_decomp_obj():
    a = Decomposition(createCube())
    a.perform_tfac()
    a.perform_PCA()
    assert len(a.PCAR2X) == len(a.sizePCA)
    assert len(a.TR2X) == len(a.sizeT)

    # test decomp save
    fname = "test_temp_atyeo.pkl"
    a.save(fname)
    b = Decomposition(None)
    b.load(fname)
    assert len(b.PCAR2X) == len(b.sizePCA)
    assert len(b.TR2X) == len(b.sizeT)
    os.remove(fname)


def test_missing_obj():
    for miss_rate in [0.1, 0.2]:
        dat = createCube()
        filter = np.random.rand(*dat.shape) > 1-miss_rate
        dat[filter] = np.nan
        a = Decomposition(dat)
        a.perform_tfac()
        a.perform_PCA()
        assert len(a.PCAR2X) == len(a.sizePCA)
        assert len(a.TR2X) == len(a.sizeT)


def test_known_rank():
    shape = (100, 80, 60)
    tFacOrig = random_cp(shape, 10, full=False)
    tOrig = tl.cp_to_tensor(tFacOrig)
    assert calcR2X(tFacOrig, tOrig) >= 1.0

    newtFac = [calcR2X(perform_CP(tOrig, r=rr), tOrig) for rr in [1,3,5,7,9]]
    assert np.all([newtFac[ii+1] > newtFac[ii]  for ii in range(len(newtFac)-1)])
    assert newtFac[0] > 0.0
    assert newtFac[-1] < 1.0

    filter = np.random.rand(*shape) > 0.8
    missT = np.copy(tOrig)
    missT[filter] = np.nan
    pass
