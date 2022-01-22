"""
This file makes all standard plots for tensor analysis
"""

import numpy as np
from matplotlib.ticker import ScalarFormatter
from .decomposition import Decomposition


def tfacr2x(ax, decomp: Decomposition):
    """
    Plot R2X of tensor decomp with more components
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
    Plot the reduced dataset size vs. (1-R2X), TFac vs. PCA
    """
    # find attributes
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
    # figure 3a in MSB
    chords_df = decomp.chordQ2X
    comps = decomp.rrs
    chords_df = chords_df.groupby('Components').agg({'R2X': ['mean', 'sem']})

    Q2Xchord = chords_df['R2X']['mean']
    Q2Xerrors = chords_df['R2X']['sem']
    ax.scatter(comps, Q2Xchord, s=10)
    ax.errorbar(comps, Q2Xchord, yerr=Q2Xerrors, fmt='none')
    ax.set_ylabel("Q2X of Imputation")
    ax.set_xlabel("Number of Components")
    ax.set_xticks([x for x in comps])
    ax.set_xticklabels([x for x in comps])
    ax.set_ylim(0, 1)

    pass


def q2xentry(ax, decomp):
    # figure 3b in MSB
    single_df = decomp.entryQ2X
    comps = decomp.rrs
    single_df = single_df.groupby(['Components']).agg(['mean', 'sem'])

    CMTFR2X = single_df['CMTF']['mean']
    CMTFErr = single_df['CMTF']['sem']
    PCAR2X = single_df['PCA']['mean']
    PCAErr = single_df['PCA']['sem']
    ax.plot(comps - 0.1, CMTFR2X, ".", label="CMTF")
    ax.plot(comps + 0.1, PCAR2X, ".", label="PCA")
    ax.errorbar(comps - 0.1, CMTFR2X, yerr=CMTFErr, fmt='none', ecolor='b')
    ax.errorbar(comps + 0.1, PCAR2X, yerr=PCAErr, fmt='none', ecolor='darkorange')
    ax.set_ylabel("Q2X of Imputation")
    ax.set_xlabel("Number of Components")
    ax.set_xticks([x for x in comps])
    ax.set_xticklabels([x for x in comps])
    ax.set_ylim(0, 1)
    ax.legend(loc=4)

    pass


def plot_weights(decomp):
    # figure 5 in MSB
    pass
