"""
Testing Decomposition
"""

import numpy as np
import os
import tensorly as tl
from tensorly.random import random_cp
from ..decomposition import impute_missing_mat, Decomposition
from ..cmtf import perform_CP, calcR2X
from tensordata.atyeo import data as atyeo
from tensordata.alter import data as alter


def test_impute_missing_mat():
    errs = []
    for _ in range(20):
        r = np.dot(np.random.rand(20, 5), np.random.rand(5, 18))
        filt = np.random.rand(20, 18) < 0.1
        rc = r.copy()
        rc[filt] = np.nan
        errs.append(np.sum((r - impute_missing_mat(rc))[filt] ** 2) / np.sum(r[filt] ** 2))
    assert np.mean(errs) < 0.1


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


def create_missingness(tensor, drop):
    idxs = np.argwhere(np.isfinite(tensor))
    ranidx = np.random.choice(idxs.shape[0], drop) 
    for idx in ranidx:
        i, j, k = idxs[idx]
        tensor[i, j, k] = np.nan


def test_entryq2x(test, drop=10, repeat=5):
    test.Q2X_entry(drop, repeat)
    print(test.entryQ2X)
    print(test.entryQ2XPCA)
    chord_drop = max(test.data.shape) // 2
    test.Q2X_chord(chord_drop, repeat)
    print(test.chordQ2X)


def create_tensors():
    shape = (10,10,10)
    tensor_1 = tl.cp_to_tensor(random_cp(shape, 3))
    test_1 =  Decomposition(tensor_1)
    tensor_2 = noise = np.random.normal(0.5,0.15, shape)
    tensor_2 = np.add(tensor_1,noise)
    create_missingness(tensor_2, drop=100)
    test_2 = Decomposition(tensor_2)
    test_atyeo = Decomposition(atyeo().tensor)
    test_alter = Decomposition(alter().tensor)
    # test_1     - (10,10,10)   //  full tensor               //   1000 values
    # test_2     - (10,10,10)   //  artifical nans & noise    //    900 values (90.0%)
    # test_atyeo - (22,12,3)    //  full tensor               //    792 values
    # test_alter - (181,22,41)  //  natural missingness       //  93577 values (57.3%)
    return test_1, test_2, test_atyeo, test_alter

