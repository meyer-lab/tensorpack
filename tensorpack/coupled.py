import xarray as xr
import numpy as np
import pandas as pd
import tensorly as tl
from tensorly.cp_tensor import CPTensor
from tqdm import tqdm
from .cmtf import calcR2X

def genSample():
    return xr.Dataset(
        data_vars=dict(
            flop=(["month", "time", "people", "state"], np.random.rand(8, 7, 6, 5)),
            turn=(["month", "time", "state"], np.random.rand(8, 7, 5)),
            river=(["month", "suit"], np.random.rand(8, 4)),
        ),
        coords=dict(
            month=["January", "February", "March", "April", "May", "June", "July", "August"],
            time=pd.date_range("2014-09-06", periods=7),
            people=["Liam", "Olivia", "Noah", "Emma", "Benjamin", "Charlotte"],
            state=["Ohio", "Tennessee", "Utah", "Virginia", "Wyoming"],
            suit=["Spade", "Heart", "Club", "Diamond"]
        ),
    )


def xr_unfold(data: xr.Dataset, mode: str):
    """ Generate the flatten array along the mode axis """
    arrs = []  # save flattened arrays
    for varname, da in data.data_vars.items():
        if mode in da.coords:
            arrs.append(tl.unfold(da.to_numpy(), list(da.coords).index(mode)))  # unfold
    return np.concatenate(arrs, axis=1)


class CoupledTensor():
    def __init__(self, data: xr.Dataset, rank):
        dd = data.to_dict()

        ncoords = {}
        for cdn in list(dd["coords"].keys()):
            ncoords[cdn] = data.coords[cdn].to_numpy()
        ncoords["*Component"] = np.arange(1, rank + 1)
        ncoords["*Data"] = list(dd["data_vars"].keys())

        ndata = {}
        for dan in list(dd["dims"].keys()):
            ndata["#" + dan] = ([dan, "*Component"], np.ones((dd["dims"][dan], rank)))
        ndata["*Weight"] = (["*Data", "*Component"], np.ones((len(dd["data_vars"]), rank)))

        self.x = xr.Dataset(
            data_vars=ndata,
            coords=ncoords,
            attrs=dict(),
        )
        self.data = data
        self.rank = rank
        self.dims = {a: dd["data_vars"][a]['dims'] for a in dd["data_vars"]}
        self.unfold = {mmode: xr_unfold(data, mmode) for mmode in list(data.coords)}


    def initialize(self, method="svd"):
        """ Initialize each mode factor matrix """
        if method == "ones":
            for mmode in list(self.data.dims):
                self.x["#"+mmode][:] = np.ones_like(self.x["#"+mmode])
        if method == "svd":
            for mmode in list(self.data.dims):
                self.x["#" + mmode][:, :min(self.rank, len(self.x[mmode]))] = np.linalg.svd(self.unfold[mmode])[0][:,
                                                                         :min(self.rank, len(self.x[mmode]))]
        self.x["*Weight"][:] = np.ones_like(self.x["*Weight"])


    def to_CPTensor(self, vars: str):
        return CPTensor((self.x["*Weight"].loc[vars, :].to_numpy(),
                            [self.x["#" + mmode].to_numpy() for mmode in self.dims[vars]]))

    def calcR2X(self, vars: str):
        return calcR2X(self.to_CPTensor(vars), self.data[vars].to_numpy())

    def reconstruct(self, vars=None):
        """ Put decomposed factors back into an xr.DataArray (when specify vars name) or and xr.Dataset """
        if vars is None:  # return the entire xr.Dataset
            ndata = {}
            R2Xs = {}
            for vars in list(self.data.data_vars):
                reconCP = self.to_CPTensor(vars)
                ndata[vars] = (self.dims[vars], reconCP.to_tensor())
                R2Xs[vars] = calcR2X(reconCP, self.data[vars].to_numpy())
            return xr.Dataset(
                data_vars=ndata,
                coords=self.data.coords,
                attrs=dict(R2X = R2Xs),
            )

        # return just one xr.DataArray
        assert vars in self.data.data_vars
        reconCP = self.to_CPTensor(vars)
        return xr.DataArray(
            data=reconCP.to_tensor(),
            coords={mmode: self.data[mmode].to_numpy() for mmode in self.dims[vars]},
            name=vars,
            attrs=dict(R2X = calcR2X(reconCP, self.data[vars].to_numpy())),
        )

    def perform_CP(self, tol=1e-6, maxiter=50, progress=True):
        """ Perform CP-like coupled tensor factorization """

        tq = tqdm(range(maxiter), disable=(not progress))
        for i in tq:
            # Solve on each mode
            tq.set_postfix(refresh=False)  #R2X=R2X, delta=R2X - R2X_last, refresh=False)

        pass




