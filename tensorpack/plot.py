"""
This file makes all standard plots for tensor analysis
"""

import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter
from .decomposition import Decomposition


def tfacr2x(ax, pos, decomp: Decomposition):
    """
    Plot R2X of tensor decomp with more components
    """
    comps = decomp.rrs
    ax[pos].scatter(comps, decomp.TR2X, s=10)
    ax[pos].set_ylabel("Tensor Fac R2X")
    ax[pos].set_xlabel("Number of Components")
    ax[pos].set_title("Variance explained by tensor decomposition")
    ax[pos].set_xticks([x for x in comps])
    ax[pos].set_xticklabels([x for x in comps])
    ax[pos].set_ylim(0, 1)
    ax[pos].set_xlim(0.5, np.amax(comps) + 0.5)


def reduction(ax, pos, decomp):
    """
    Plot the reduced dataset size vs. (1-R2X), TFac vs. PCA
    """
    CPR2X, PCAR2X, sizeTfac, sizePCA = np.asarray(decomp.TR2X), np.asarray(decomp.PCAR2X), decomp.sizeT, decomp.sizePCA
    ax[pos].set_xscale("log", base=2)
    ax[pos].plot(sizeTfac, 1.0 - CPR2X, ".", label="TFac")
    ax[pos].plot(sizePCA, 1.0 - PCAR2X, ".", label="PCA")
    ax[pos].set_ylabel("Normalized Unexplained Variance")
    ax[pos].set_xlabel("Size of Reduced Data")
    ax[pos].set_title("Data reduction, TFac vs. PCA")
    ax[pos].set_ylim(bottom=0.0)
    ax[pos].xaxis.set_major_formatter(ScalarFormatter())
    ax[pos].legend()


def q2xchord(ax, pos, decomp):
    chords_df = decomp.chordQ2X
    comps = decomp.rrs
    chords_df = pd.DataFrame(decomp.chordQ2X).T
    chords_df.index = comps
    chords_df['mean'] = chords_df.mean(axis=1)
    chords_df['sem'] = chords_df.sem(axis=1)

    Q2Xchord = chords_df['mean']
    Q2Xerrors = chords_df['sem']
    ax[pos].scatter(comps, Q2Xchord, s=10)
    ax[pos].errorbar(comps, Q2Xchord, yerr=Q2Xerrors, fmt='none')
    ax[pos].set_ylabel("Q2X of Imputation")
    ax[pos].set_xlabel("Number of Components")
    ax[pos].set_xticks([x for x in comps])
    ax[pos].set_xticklabels([x for x in comps])
    ax[pos].set_ylim(bottom=0.0, top=1.0)


def q2xentry(ax, pos, decomp, methodname = "CP"):
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
    ax[pos].plot(comps - 0.1, TR2X, ".", label=methodname)
    ax[pos].plot(comps + 0.1, PCAR2X, ".", label="PCA")
    ax[pos].errorbar(comps - 0.1, TR2X, yerr=TErr, fmt='none', ecolor='b')
    ax[pos].errorbar(comps + 0.1, PCAR2X, yerr=PCAErr, fmt='none', ecolor='darkorange')
    ax[pos].set_ylabel("Q2X of Imputation")
    ax[pos].set_xlabel("Number of Components")
    ax[pos].set_xticks([x for x in comps])
    ax[pos].set_xticklabels([x for x in comps])
    ax[pos].set_ylim(0, 1)
    ax[pos].legend(loc=4)


def plot_weights(decomp):
    # figure 5 in MSB
    pass
