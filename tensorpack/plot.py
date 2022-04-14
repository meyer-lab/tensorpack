"""
This file makes all standard plots for tensor analysis. Requires a Decomposition object after running relevant values.
"""

import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter
from .decomposition import Decomposition
from tensorpack import perform_CP


def tfacr2x(ax, decomp:Decomposition):
    """
    Plots R2X for tensor factorizations for all components up to decomp.max_rr.

    Parameters
    ----------
    ax : axis object
        Plot information for a subplot of figure f. See getSetup() in tensorpack.test.common.py for more detail.
    decomp : Decomposition
        Takes a Decomposition object that has successfully run decomp.perform_tfac().
    """
    comps = decomp.rrs
    ax.scatter(comps, decomp.TR2X, s=10)
    ax.set_ylabel("Tensor Fac R2X")
    ax.set_xlabel("Number of Components")
    ax.set_title("Variance explained by tensor decomposition")
    ax.set_xticks([x for x in comps])
    ax.set_xticklabels([x for x in comps])
    ax.set_ylim(0, 1)
    ax.set_xlim(0.5, np.amax(comps) + 0.5)


def reduction(ax, decomp):
    """
    Plots size reduction for tensor factorization versus PCA for all components up to decomp.max_rr.

    Parameters
    ----------
    ax : axis object
        Plot information for a subplot of figure f.
    decomp : Decomposition
        Takes a Decomposition object that has successfully run decomp.perform_tfac() and decomp.perform_PCA().
    """
    CPR2X, PCAR2X, sizeTfac, sizePCA = np.asarray(decomp.TR2X), np.asarray(decomp.PCAR2X), decomp.sizeT, decomp.sizePCA
    ax.set_xscale("log", base=2)
    ax.plot(sizeTfac, 1.0 - CPR2X, ".", label="TFac")
    ax.plot(sizePCA, 1.0 - PCAR2X, ".", label="PCA")
    ax.set_ylabel("Normalized Unexplained Variance")
    ax.set_xlabel("Size of Reduced Data")
    ax.set_title("Data reduction, TFac vs. PCA")
    ax.set_ylim(bottom=0.0)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.legend()


def q2xchord(ax, decomp):
    """
    Plots Q2X for tensor factorization when removing chords from a single mode for all components up to decomp.max_rr.
    Requires multiple runs to generate error bars.

    Parameters
    ----------
    ax : axis object
        Plot information for a subplot of figure f.
    decomp : Decomposition
        Takes a Decomposition object that has successfully run decomp.Q2X_chord().
    """
    chords_df = decomp.chordQ2X
    comps = decomp.rrs
    chords_df = pd.DataFrame(decomp.chordQ2X).T
    chords_df.index = comps
    chords_df['mean'] = chords_df.mean(axis=1)
    chords_df['sem'] = chords_df.sem(axis=1)

    Q2Xchord = chords_df['mean']
    Q2Xerrors = chords_df['sem']
    ax.scatter(comps, Q2Xchord, s=10)
    ax.errorbar(comps, Q2Xchord, yerr=Q2Xerrors, fmt='none')
    ax.set_ylabel("Q2X of Chord Imputation")
    ax.set_xlabel("Number of Components")
    ax.set_xticks([x for x in comps])
    ax.set_xticklabels([x for x in comps])
    ax.set_ylim(bottom=0.0, top=1.0)


def q2xentry(ax, decomp, methodname = "CP"):
    """
    Plots Q2X for tensor factorization versus PCA when removing entries for all components up to decomp.max_rr.
    Requires multiple runs to generate error bars.

    Parameters
    ----------
    ax : axis object
        Plot information for a subplot of figure f.
    decomp : Decomposition
        Takes a Decomposition object that has successfully run decomp.entry().
    methodname : str
        Allows for proper tensor method when naming graph axes. 
    """
    entry_df = pd.DataFrame(decomp.entryQ2X).T
    entrypca_df = pd.DataFrame(decomp.entryQ2XPCA).T
    comps = decomp.rrs

    entry_df.index = comps
    entry_df['mean'] = entry_df.mean(axis=1)
    entry_df['sem'] = entry_df.sem(axis=1)
    entrypca_df.index = comps
    entrypca_df['mean'] = entrypca_df.mean(axis=1)
    entrypca_df['sem'] = entrypca_df.sem(axis=1)

    TR2X = entry_df['mean']
    TErr = entry_df['sem']
    PCAR2X = entrypca_df['mean']
    PCAErr = entrypca_df['sem']
    ax.plot(comps - 0.05, TR2X, ".", label=methodname)
    ax.plot(comps + 0.05, PCAR2X, ".", label="PCA")
    ax.errorbar(comps - 0.05, TR2X, yerr=TErr, fmt='none', ecolor='b')
    ax.errorbar(comps + 0.05, PCAR2X, yerr=PCAErr, fmt='none', ecolor='darkorange')
    ax.set_ylabel("Q2X of Entry Imputation")
    ax.set_xlabel("Number of Components")
    ax.set_xticks([x for x in comps])
    ax.set_xticklabels([x for x in comps])
    ax.set_ylim(0, 1)
    ax.legend(loc=4)


def plot_weight_mode(ax, factor, labels=False, title = ""):
    """
    Plots heatmaps for a single mode factors.

    Parameters
     ----------
    ax : axis object
        Plot information for a subplot of figure f.
    factor: numpy array
        Factorized mode
    labels: list of string or False
        Labels for each of the elements
    title" String
        Figure title
    """
    rank = np.shape(factor)[1]
    components = [str(ii + 1) for ii in range(rank)]
    facs = pd.DataFrame(factor, columns=[f"Cmp. {i}" for i in np.arange(1, rank + 1)],
                        index=labels if labels is not False else list(range(np.shape(factor)[0])))

    sns.heatmap(facs, cmap="PiYG", center=0, xticklabels=components, yticklabels=labels, cbar=True, vmin=-1.0,
                vmax=1.0, ax=ax)

    ax.set_xlabel("Components")
    ax.set_title(title)


