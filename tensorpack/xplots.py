import pandas as pd
import xarray as xr
import numpy as np
from .cmtf import perform_CP, calcR2X

from matplotlib import gridspec, pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import scipy.cluster.hierarchy as sch

def xplot_R2X(data:xr.DataArray, top_rank=12, ax=None, method=perform_CP):
    """ Plot increasing rank R2X for CP """
    assert isinstance(data, xr.DataArray) or isinstance(data, np.ndarray)
    ranks = np.arange(1, min(np.min(data.shape), top_rank)+1)
    R2Xs = []

    for r in ranks:
        cp = method(data.to_numpy(), r)
        R2Xs.append(calcR2X(cp, tIn=data.to_numpy()))

    plt_indep = ax is None
    if plt_indep:
        f = plt.figure(figsize=(5, 4))
        gs = gridspec.GridSpec(1, 1, wspace=0.5)
        ax = plt.subplot(gs[0])

    ax = sns.lineplot(x=ranks, y=R2Xs, ax=ax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Number of components")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("R2X")
    ax.set_title("Variance Explained by CP")

    return (f, ax) if plt_indep else ax


def xplot_components(data:xr.DataArray, rank: int, reorder=[]):
    """ Plot the heatmaps of each components from an xarray-formatted data. """
    cp = perform_CP(data.to_numpy(), rank)
    ddims = len(data.dims)
    axes_names = list(data.dims)

    factors = [pd.DataFrame(cp[1][rr],
                            columns=[f"Cmp. {i}" for i in np.arange(1, rank + 1)],
                            index=data.coords[axes_names[rr]].values  \
                                if len(data.coords[axes_names[rr]].coords) <= 1 \
                                else [" ".join(ss) for ss in data.coords["Analyte"].values])
               for rr in range(ddims)]

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