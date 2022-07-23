import pandas as pd
import xarray as xr
import numpy as np
from .cmtf import perform_CP
from matplotlib import gridspec, pyplot as plt
import seaborn as sns
from .decomposition import Decomposition

def xplot_reduction(data:xr.DataArray):
    """ Plot increasing rank R2X and data reduction of PCA/CP/Tucker """
    pass


def xplot_components(data:xr.DataArray, rank: int):
    """ Plot the heatmaps of each components from an xarray-formatted data. """
    cp = perform_CP(data.to_numpy(), rank)
    ddims = len(data.coords)
    axes_names = list(data.coords)

    factors = [pd.DataFrame(cp[1][rr], columns=[f"Cmp. {i}" for i in np.arange(1, rank + 1)],
                            index=data.coords[axes_names[rr]]) for rr in range(ddims)]

    f = plt.figure(figsize=(5*ddims, 6))
    gs = gridspec.GridSpec(1, ddims, wspace=0.5)
    axes = [plt.subplot(gs[rr]) for rr in range(ddims)]
    comp_labels = [str(ii + 1) for ii in range(rank)]

    for rr in range(ddims):
        sns.heatmap(factors[rr], cmap="PiYG", center=0, xticklabels=comp_labels, yticklabels=factors[rr].index,
                    cbar=True, vmin=-1.0, vmax=1.0, ax=axes[rr])
        axes[rr].set_xlabel("Components")
        axes[rr].set_title(axes_names[rr])
    return f