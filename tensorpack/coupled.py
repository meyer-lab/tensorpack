import xarray as xr
import numpy as np
import pandas as pd
import tensorly as tl
from tensorly.cp_tensor import CPTensor
from numpy.linalg import lstsq, norm
from tqdm import tqdm

def genSample():
    return xr.Dataset(
        data_vars=dict(
            vlop=(["month", "time", "people", "state"], np.random.rand(8, 7, 6, 5)),
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
    for _, da in data.data_vars.items():
        if mode in da.coords:
            arrs.append(tl.unfold(da.to_numpy(), list(da.coords).index(mode)))  # unfold
    return np.concatenate(arrs, axis=1)


def calcR2X_TnB(tIn, tRecon):
    """ Calculate the top and bottom part of R2X formula separately """
    tMask = np.isfinite(tIn)
    tIn = np.nan_to_num(tIn)
    vTop = norm(tRecon * tMask - tIn) ** 2.0
    vBottom = norm(tIn) ** 2.0
    return vTop, vBottom


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
        self.data_vars = list(self.data.data_vars)
        self.data_coords = list(self.data.dims)
        self.dims = {a: dd["data_vars"][a]['dims'] for a in dd["data_vars"]}
        self.unfold = {mmode: xr_unfold(data, mmode) for mmode in list(data.coords)}


    def initialize(self, method="svd"):
        """ Initialize each mode factor matrix """
        if method == "ones":
            for mmode in self.data_coords:
                self.x["#"+mmode][:] = np.ones_like(self.x["#"+mmode])
        if method == "svd":
            ## TODO: add missing data handling here
            for mmode in self.data_coords:
                self.x["#" + mmode][:, :min(self.rank, len(self.x[mmode]))] = np.linalg.svd(self.unfold[mmode])[0][:,
                                                                         :min(self.rank, len(self.x[mmode]))]
        self.x["*Weight"][:] = np.ones_like(self.x["*Weight"])


    def to_CPTensor(self, dvars: str):
        """ Return a CPTensor object that is the factorized version of dvars """
        assert dvars in self.data_vars
        return CPTensor((self.x["*Weight"].loc[dvars, :].to_numpy(),
                            [self.x["#" + mmode].to_numpy() for mmode in self.dims[dvars]]))

    def calcR2X(self, dvars=None):
        """ Calculate the R2X of dvars decomposition. If dvars not provide, calculate the overall R2X"""
        if dvars is None:    # find overall R2X
            vTop, vBottom = 0.0, 0.0
            for dvars in self.data_vars:
                top, bot = calcR2X_TnB(self.data[dvars].to_numpy(), self.to_CPTensor(dvars).to_tensor())
                vTop += top
                vBottom += bot
            return 1.0 - vTop / vBottom

        assert dvars in self.data_vars
        vTop, vBottom = calcR2X_TnB(self.data[dvars].to_numpy(), self.to_CPTensor(dvars).to_tensor())
        return 1.0 - vTop / vBottom

    def reconstruct(self, dvars=None):
        """ Put decomposed factors back into an xr.DataArray (when specify dvars name) or and xr.Dataset """
        if dvars is None:  # return the entire xr.Dataset
            ndata = {}
            R2Xs = {}
            for dvars in self.data_vars:
                ndata[dvars] = (self.dims[dvars], self.to_CPTensor(dvars).to_tensor())
                R2Xs[dvars] = self.calcR2X(dvars)     # a bit redundant, but more beautiful
            return xr.Dataset(
                data_vars=ndata,
                coords=self.data.coords,
                attrs=dict(R2X = R2Xs),
            )

        # return just one xr.DataArray
        assert dvars in self.data_vars
        return xr.DataArray(
            data=self.to_CPTensor(dvars).to_tensor(),
            coords={mmode: self.data[mmode].to_numpy() for mmode in self.dims[dvars]},
            name=dvars,
            attrs=dict(R2X = self.calcR2X(dvars)),
        )

    def khatri_rao(self, mode: str):
        """ Find the Khatri-Rao product on a certain mode after concatenation """
        assert mode in self.data_coords
        arrs = []  # save kr-ed arrays
        for dvars in self.data_vars:
            if mode in self.dims[dvars]:
                recon = tl.tenalg.khatri_rao([self.x["#"+mmode].to_numpy() for mmode in self.dims[dvars] if mmode != mode])
                arrs.append(recon * self.x["*Weight"].loc[dvars].to_numpy())    # put weights back to kr
        return np.concatenate(arrs, axis=0)


    def perform_CP(self, tol=1e-7, maxiter=100, progress=True):
        """ Perform CP-like coupled tensor factorization """
        old_R2X = -np.inf
        tq = tqdm(range(maxiter), disable=(not progress))
        for i in tq:
            # Solve on each mode
            for mmode in self.data_coords:
                sol = lstsq(self.khatri_rao(mmode), self.unfold[mmode].T, rcond=None)[0].T
                for dvars in self.data_vars:
                    if mmode in self.dims[dvars]:
                        self.x["*Weight"].loc[dvars] *= norm(sol, axis=0)
                self.x["#"+mmode][:] = sol / norm(sol, axis=0)

            current_R2X = self.calcR2X()
            #print([self.calcR2X(dvars) for dvars in self.data_vars])
            tq.set_postfix(refresh=False, R2X=current_R2X, delta=current_R2X-old_R2X)
            if np.abs(current_R2X-old_R2X) < tol:
                break
            old_R2X = current_R2X


    def plot_factors(self, reorder=[]):
        """ Plot the factors of each mode """
        from matplotlib import gridspec, pyplot as plt
        import seaborn as sns
        import scipy.cluster.hierarchy as sch
        from .xplots import reorder_table

        ddims = len(self.data_coords)
        factors = [self.x['#'+mode].to_pandas() for mode in self.data_coords]

        for r_ax in reorder:
            if isinstance(r_ax, int):
                assert r_ax < ddims
                factors[r_ax] = reorder_table(factors[r_ax])
            elif isinstance(r_ax, str):
                assert r_ax in self.data_coords
                rr = self.data_coords.index(r_ax)
                factors[rr] = reorder_table(factors[rr])
            else:
                raise TypeError("reorder only takes a list of int's or str's.")

        f = plt.figure(figsize=(5 * ddims, 6))
        gs = gridspec.GridSpec(1, ddims, wspace=0.5)
        axes = [plt.subplot(gs[rr]) for rr in range(ddims)]
        comp_labels = [str(ii + 1) for ii in range(self.rank)]

        for rr in range(ddims):
            sns.heatmap(factors[rr], cmap="PiYG", center=0, xticklabels=comp_labels, yticklabels=factors[rr].index,
                        cbar=True, vmin=-1.0, vmax=1.0, ax=axes[rr])
            axes[rr].set_xlabel("Components")
            axes[rr].set_ylabel(None)
            axes[rr].set_title(self.data_coords[rr].capitalize())
        return f, axes



