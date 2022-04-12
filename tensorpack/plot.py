"""
This file makes all standard plots for tensor analysis. Requires a Decomposition object after running relevant values.
"""

import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter
from .decomposition import Decomposition
from tensorpack import perform_CP
import seaborn as sns
from matplotlib import gridspec, pyplot as plt


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


def plot_weight_mode(ax, factor, labels, showylabel = True, title = ""):
    """
    Plots heatmaps for each input dimension by component.

    Parameters
     ----------
    ax : axis object
        Plot information for a subplot of figure f.
    tensor : numpy ndarray
        Takes a tensor to perform CP on.
    axes : tuple of 3 lists
        Allows for naming graph axes.
    """
    rank = np.shape(factor)[1]
    components = [str(ii + 1) for ii in range(rank)]
    facs = pd.DataFrame(factor, columns=[f"Cmp. {i}" for i in np.arange(1, rank + 1)], index=labels)

    yticklabels = labels if showylabel else False
    sns.heatmap(facs, cmap="PiYG", center=0, xticklabels=components, yticklabels=yticklabels, cbar=True, vmin=-1.0,
                vmax=1.0, ax=ax)

    ax.set_xlabel("Components")
    ax.set_title(title)


def plot_weights(ax, tfac, axes):
    """
    Plots heatmaps for each input dimension by component.

    Parameters
    ----------
    ax : axis object
        Plot information for a subplot of figure f.
    tensor : numpy ndarray 
        Takes a tensor to perform CP on.
    axes : tuple of 3 lists
        Allows for naming graph axes. 
    """
    subjects, receptors, antigens = axes

    components =  [str(ii + 1) for ii in range(tfac.rank)]

    subs = pd.DataFrame(tfac.factors[0], columns=[f"Cmp. {i}" for i in np.arange(1, tfac.rank + 1)], index=subjects)
    rec = pd.DataFrame(tfac.factors[1], columns=[f"Cmp. {i}" for i in np.arange(1, tfac.rank + 1)], index=receptors)
    ant = pd.DataFrame(tfac.factors[2], columns=[f"Cmp. {i}" for i in np.arange(1, tfac.rank + 1)], index=antigens)
    
    gs = gridspec.GridSpec(1, 3, wspace=0.5)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])

    plot_weight_mode(ax1, tfac.factors[0], subjects, showylabel=False, title="Subjects")
    plot_weight_mode(ax2, tfac.factors[1], receptors, showylabel=True, title="Subjects")
    plot_weight_mode(ax3, tfac.factors[2], antigens, showylabel=True, title="Subjects")

    sns.heatmap(subs, cmap="PiYG", center=0, xticklabels=components, cbar=True, vmin=-1.0, vmax=1.0, ax=ax1)
    sns.heatmap(rec, cmap="PiYG", center=0, xticklabels=components, yticklabels=receptors, cbar=False, vmin=-1.0, vmax=1.0, ax=ax2)
    sns.heatmap(ant, cmap="PiYG", center=0, xticklabels=components, yticklabels=antigens, cbar=False, vmin=-1.0, vmax=1.0, ax=ax3)

    ax1.set_xlabel("Components")
    ax1.set_title("Subjects")
    ax2.set_xlabel("Components")
    ax2.set_title("Receptors")
    ax3.set_xlabel("Components")
    ax3.set_title("Antigens")


