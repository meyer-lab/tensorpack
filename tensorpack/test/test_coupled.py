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
    for r in np.arange(3,9,2):
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
    for r in np.arange(3,9,2):
        cpd = CoupledTensor(data, r)
        cpd.initialize("nmf")
        cpd.fit(nonneg=True)
        R2X = cpd.R2X()
        assert oldR2X <= R2X + 1e-2, f"r = {r}: oldR2X = {oldR2X}, but newR2X = {R2X}"
        oldR2X = R2X
        assert np.all(np.array([cpd.R2X(dvar) for dvar in cpd.dvars]) > 0.7)