"""
Testing Decomposition
"""

import os
import numpy as np
import tensorly as tl
from tensorly.random import random_cp
from ..decomposition import Decomposition
from ..cmtf import perform_CP, calcR2X
from tensordata.atyeo import data as atyeo
from ..SVD_impute import IterativeSVD


def test_impute_missing_mat():
    errs = []
    for _ in range(20):
        r = np.dot(np.random.rand(20, 5), np.random.rand(5, 18))
        filt = np.random.rand(20, 18) < 0.1
        rc = r.copy()
        rc[filt] = np.nan
        imp = IterativeSVD(rank=1, random_state=1).fit_transform(rc)
        errs.append(np.sum((r - imp)[filt] ** 2) / np.sum(r[filt] ** 2))
    assert np.mean(errs) < 0.03


def test_decomp_obj():
    a = Decomposition(atyeo().tensor)
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
        dat = random_cp((20, 10, 8), 5, full=True)
        filter = np.random.rand(*dat.shape) > 1 - miss_rate
        dat[filter] = np.nan
        a = Decomposition(dat)
        a.perform_tfac()
        a.perform_PCA()
        assert len(a.PCAR2X) == len(a.sizePCA)
        assert len(a.TR2X) == len(a.sizeT)


def test_known_rank():
    shape = (50, 40, 30)
    tFacOrig = random_cp(shape, 10, full=False)
    tOrig = tl.cp_to_tensor(tFacOrig)
    assert calcR2X(tFacOrig, tOrig) >= 1.0

    newtFac = [calcR2X(perform_CP(tOrig, r=rr), tOrig) for rr in [1, 3, 5, 7, 9]]
    assert np.all([newtFac[ii + 1] > newtFac[ii] for ii in range(len(newtFac) - 1)])
    assert newtFac[0] > 0.0
    assert newtFac[-1] < 1.0
