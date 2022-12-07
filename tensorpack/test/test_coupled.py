"""
Unit test file.
"""

import numpy as np
import pandas as pd
import xarray as xr
from ..coupled import CoupledTensor

def genSample(missing=0.0):
    dasset = np.random.rand(8, 7, 6, 5)
    dliab = np.random.rand(8, 7, 5)
    dequity = np.random.rand(8, 4)
    if missing > 0.0:
        dasset[np.random.rand(*dasset.shape) < missing] = np.nan
        dliab[np.random.rand(*dliab.shape) < missing] = np.nan
        dequity[np.random.rand(*dequity.shape) < missing] = np.nan

    return xr.Dataset(
        data_vars=dict(
            asset=(["month", "time", "people", "state"], dasset),
            liability=(["month", "time", "state"], dliab),
            equity=(["month", "suit"], dequity),
        ),
        coords=dict(
            month=["January", "February", "March", "April", "May", "June", "July", "August"],
            time=pd.date_range("2014-09-06", periods=7),
            people=["Liam", "Olivia", "Noah", "Emma", "Benjamin", "Charlotte"],
            state=["Ohio", "Tennessee", "Utah", "Virginia", "Wyoming"],
            suit=["Spade", "Heart", "Club", "Diamond"]
        ),
    )

def test_coupled_svd():
    data = genSample()
    oldR2X = -np.inf
    for r in [3,5,7,9]:
        cpd = CoupledTensor(data, r)
        cpd.initialize("svd")
        cpd.fit()
        R2X = cpd.R2X()
        assert oldR2X <= R2X + 1e-5, f"r = {r}: oldR2X = {oldR2X}, but newR2X = {R2X}"
        oldR2X = R2X
        assert np.all(np.array([cpd.R2X(dvar) for dvar in cpd.dvars]) > 0.7)

def test_coupled_nonneg():
    data = genSample()
    oldR2X = -np.inf
    for r in [3,5,7,9]:
        cpd = CoupledTensor(data, r)
        cpd.initialize("nmf")
        cpd.fit(nonneg=True)
        R2X = cpd.R2X()
        assert oldR2X <= R2X + 1e-2, f"r = {r}: oldR2X = {oldR2X}, but newR2X = {R2X}"
        oldR2X = R2X
        assert np.all(np.array([cpd.R2X(dvar) for dvar in cpd.dvars]) > 0.7)

def test_randomized_svd():
    data = xr.Dataset(
        data_vars=dict(
            Adam=(["Patient", "Box", "Gene", "Visit"], np.random.rand(148, 2, 7139, 6)),
            Brendan=(["Patient", "Cytokine"], np.random.rand(148, 38)),
        ))
    cp = CoupledTensor(data, 2)
    blank_R2X = cp.R2X()
    cp.initialize("randomized_svd")
    init_R2X = cp.R2X()
    assert init_R2X > blank_R2X
    cp.fit(maxiter=2, verbose=True)
    fit_R2X = cp.R2X()
    assert fit_R2X > init_R2X
