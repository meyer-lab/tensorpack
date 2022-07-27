import pandas as pd
import xarray as xr
import numpy as np
from .cmtf import perform_CP
from matplotlib import gridspec, pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
from .decomposition import Decomposition

def xplot_reduction(data:xr.DataArray):
    """ Plot increasing rank R2X and data reduction of PCA/CP/Tucker """
    pass


def xplot_components(data:xr.DataArray, rank: int, reorder=[]):
    """ Plot the heatmaps of each components from an xarray-formatted data. """
    cp = perform_CP(data.to_numpy(), rank)
    ddims = len(data.coords)
    axes_names = list(data.coords)

    factors = [pd.DataFrame(cp[1][rr], columns=[f"Cmp. {i}" for i in np.arange(1, rank + 1)],
                            index=data.coords[axes_names[rr]]) for rr in range(ddims)]

    for r_ax in reorder:
        if isinstance(r_ax, int):
            assert r_ax < ddims
            factors[r_ax] = reorder_table(factors[r_ax])
        elif isinstance(r_ax, str):
            assert r_ax in axes_names
            rr = axes_names.index(r_ax)
            factors[rr] = reorder_table(factors[rr])

    f = plt.figure(figsize=(5*ddims, 6))
    gs = gridspec.GridSpec(1, ddims, wspace=0.5)
    axes = [plt.subplot(gs[rr]) for rr in range(ddims)]
    comp_labels = [str(ii + 1) for ii in range(rank)]

    for rr in range(ddims):
        sns.heatmap(factors[rr], cmap="PiYG", center=0, xticklabels=comp_labels, yticklabels=factors[rr].index,
                    cbar=True, vmin=-1.0, vmax=1.0, ax=axes[rr])
        axes[rr].set_xlabel("Components")
        axes[rr].set_title(axes_names[rr])
    return f, axes


def reorder_table(df):
    """
    Reorder a table's rows using hierarchical clustering.
    Parameters:
        df (pandas.DataFrame): data to be clustered; rows are treated as samples
            to be clustered
    Returns:
        df (pandas.DataFrame): data with rows reordered via heirarchical
            clustering
    """
    y = sch.linkage(df.to_numpy(), method='centroid')
    index = sch.dendrogram(y, orientation='right', no_plot=True)['leaves']
    return df.iloc[index, :]