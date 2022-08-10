import xarray as xr
import numpy as np
import tensorly as tl
from tensorly.cp_tensor import CPTensor
from numpy.linalg import lstsq, norm
from tqdm import tqdm
from .SVD_impute import IterativeSVD
from .cmtf import censored_lstsq


def xr_unfold(data: xr.Dataset, mode: str):
    """ Generate the flatten array along the mode axis """
    arrs = []  # save flattened arrays
    for _, da in data.data_vars.items():
        if mode in da.dims:
            arrs.append(tl.unfold(da.to_numpy(), list(da.dims).index(mode)))  # unfold
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

        self.data = data
        self.rank = rank
        self.dvars = list(self.data.data_vars)
        self.modes = list(self.data.dims)

        ncoords = {}
        ndata = {}
        for mode in self.modes:
            ncoords[mode] = data.coords[mode].to_numpy()
            ndata["_" + mode] = ([mode, "_Component_"], np.ones((dd["dims"][mode], rank)))
        ncoords["_Component_"] = np.arange(1, rank + 1)
        ncoords["_Data_"] = self.dvars
        ndata["_Weight_"] = (["_Data_", "_Component_"], np.ones((len(self.dvars), rank)))

        self.x = xr.Dataset(
            data_vars=ndata,
            coords=ncoords,
            attrs=dict(),
        )

        self.dims = {a: dd["data_vars"][a]['dims'] for a in dd["data_vars"]}    # same idea as self.dvar_to_mode
        self.mode_to_dvar = {mode: tuple(dvar for dvar in self.dvars if mode in self.dims[dvar]) for mode in self.modes}
        self.unfold = {mode: xr_unfold(data, mode) for mode in self.modes}


    def initialize(self, method="svd", verbose=False):
        """ Initialize each mode factor matrix """
        if method == "ones":
            for mmode in self.modes:
                self.x["_"+mmode][:] = np.ones_like(self.x["_"+mmode])
        if method == "svd":
            import time
            for mmode in self.modes:
                start_time = time.time()
                unfold = self.unfold[mmode].copy()
                ncol = min(self.rank, len(self.x[mmode]))
                if np.sum(~np.isfinite(unfold)) > 0:
                    si = IterativeSVD(ncol, max_iters=50)
                    unfold = si.fit_transform(unfold[:, ~np.all(np.isnan(unfold), axis=0)])
                    ncol = min(ncol, unfold.shape[1])   # case where num col in reduced unfold is even smaller
                self.x["_"+mmode][:, :ncol] = np.linalg.svd(unfold)[0][:,:ncol]
                if verbose:
                    print(f"{mmode} SVD initialization: done in {time.time() - start_time}")
        self.x["_Weight_"][:] = np.ones_like(self.x["_Weight_"])


    def to_CPTensor(self, dvar: str):
        """ Return a CPTensor object that is the factorized version of dvar """
        assert dvar in self.dvars
        return CPTensor((self.x["_Weight_"].loc[dvar, :].to_numpy(),
                            [self.x["_"+mmode].to_numpy() for mmode in self.dims[dvar]]))

    def calcR2X(self, dvar=None):
        """ Calculate the R2X of dvar decomposition. If dvar not provide, calculate the overall R2X"""
        if dvar is None:    # find overall R2X
            vTop, vBottom = 0.0, 0.0
            for dvar in self.dvars:
                top, bot = calcR2X_TnB(self.data[dvar].to_numpy(), self.to_CPTensor(dvar).to_tensor())
                vTop += top
                vBottom += bot
            return 1.0 - vTop / vBottom

        assert dvar in self.dvars
        vTop, vBottom = calcR2X_TnB(self.data[dvar].to_numpy(), self.to_CPTensor(dvar).to_tensor())
        return 1.0 - vTop / vBottom

    def reconstruct(self, dvar=None):
        """ Put decomposed factors back into an xr.DataArray (when specify dvar name) or and xr.Dataset """
        if dvar is None:  # return the entire xr.Dataset
            ndata = {}
            R2Xs = {}
            for dvar in self.dvars:
                ndata[dvar] = (self.dims[dvar], self.to_CPTensor(dvar).to_tensor())
                R2Xs[dvar] = self.calcR2X(dvar)     # a bit redundant, but more beautiful
            return xr.Dataset(
                data_vars=ndata,
                coords=self.data.coords,
                attrs=dict(R2X = R2Xs),
            )

        # return just one xr.DataArray
        assert dvar in self.dvars
        return xr.DataArray(
            data=self.to_CPTensor(dvar).to_tensor(),
            coords={mmode: self.data[mmode].to_numpy() for mmode in self.dims[dvar]},
            name=dvar,
            attrs=dict(R2X = self.calcR2X(dvar)),
        )

    def khatri_rao(self, mode: str):
        """ Find the Khatri-Rao product on a certain mode after concatenation """
        assert mode in self.modes
        arrs = []  # save kr-ed arrays
        for dvar in self.mode_to_dvar[mode]:
            recon = tl.tenalg.khatri_rao([self.x["_"+mmode].to_numpy() for mmode in self.dims[dvar] if mmode != mode])
            arrs.append(recon * self.x["_Weight_"].loc[dvar].to_numpy())    # put weights back to kr
        return np.concatenate(arrs, axis=0)


    def perform_CP(self, tol=1e-7, maxiter=500, progress=True, verbose=False):
        """ Perform CP-like coupled tensor factorization """
        old_R2X = -np.inf
        tq = tqdm(range(maxiter), disable=(not progress))

        # missing value handling
        uniqueInfo = {}
        containMissing = {}
        for mmode in self.modes:
            containMissing[mmode] = np.sum(~np.isfinite(self.unfold[mmode])) > 0
            uniqueInfo[mmode] = np.unique(np.isfinite(self.unfold[mmode].T), axis=1, return_inverse=True)

        for i in tq:
            # Solve on each mode
            for mmode in self.modes:
                if containMissing[mmode]:
                    sol = censored_lstsq(self.khatri_rao(mmode), self.unfold[mmode].T, uniqueInfo[mmode])
                else:
                    sol = lstsq(self.khatri_rao(mmode), self.unfold[mmode].T, rcond=None)[0].T
                for dvar in self.mode_to_dvar[mmode]:
                    self.x["_Weight_"].loc[dvar] *= norm(sol, axis=0)
                self.x["_"+mmode][:] = sol / norm(sol, axis=0)

            current_R2X = self.calcR2X()
            if verbose:
                print(f"R2Xs at {i}: {[self.calcR2X(dvar) for dvar in self.dvars]}")
            tq.set_postfix(refresh=False, R2X=current_R2X, delta=current_R2X-old_R2X)
            if np.abs(current_R2X-old_R2X) < tol:
                break
            old_R2X = current_R2X

    ## TODO: flip signs to make most components positive
    ## TODO: sort components based on (1) weights (2) clustering


    def plot_factors(self, dvar=None, reorder=[], normalize=True):
        """ Plot the factors of each mode. If dvar not specified, plot all """
        from matplotlib import gridspec, pyplot as plt
        import seaborn as sns
        from .xplots import reorder_table

        modes = self.modes if dvar is None else self.dims[dvar]
        ddims = len(modes)
        factors = [self.x["_"+mode].to_pandas() for mode in modes]

        if normalize:
            for i in range(len(factors)):
                factors[i] /= np.linalg.norm(factors[i], ord=np.inf, axis=0)

        for r_ax in reorder:
            if isinstance(r_ax, int):
                assert r_ax < ddims
                factors[r_ax] = reorder_table(factors[r_ax])
            elif isinstance(r_ax, str):
                assert r_ax in modes
                rr = modes.index(r_ax)
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
            axes[rr].set_title(modes[rr])
        return f, axes

