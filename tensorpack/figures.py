import seaborn as sns

"""
pickle file contains two arrays:
pca_rs = [x1, x2, ... ,xr]
tensor_rs = [y1, y2, ... ,yr]
"""

import numpy as np
import pandas as pd
from statsmodels.multivariate.pca import PCA
from .figureCommon import subplotLabel, getSetup
from matplotlib.ticker import ScalarFormatter


def plot_r2x(ax, CMTFR2X, comps):
    # figure 2a in MSB
    ax.scatter(comps, CMTFR2X, s=10)
    ax.set_ylabel("CMTF R2X")
    ax.set_xlabel("Number of Components")
    ax.set_xticks([x for x in comps])
    ax.set_xticklabels([x for x in comps])
    ax.set_ylim(0, 1)
    ax.set_xlim(0.5, np.amax(comps) + 0.5)

    pass

def plot_reduction(ax, CMTFR2X, PCAR2X, sizeTfac, sizePCA):
    # figure 2b in MSB
    ax.set_xscale("log", base=2)
    ax.plot(sizeTfac, 1.0 - CMTFR2X, ".", label="CMTF")
    ax.plot(sizePCA, 1.0 - PCAR2X, ".", label="PCA")
    ax.set_ylabel("Normalized Unexplained Variance")
    ax.set_xlabel("Size of Reduced Data")
    ax.set_ylim(bottom=0.0)
    ax.set_xlim(2 ** 8, 2 ** 12)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.legend()

    pass

def plot_q2x_chord(ax, csv_file, comps):
    # figure 3a in MSB
    chords_df = pd.read_csv(csv_file)
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

def plot_q2x_entries(ax, csv_file, comps):
    # figure 3b in MSB
    single_df = pd.read_csv(csv_file)
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

def plot_weights(pickle_file):
    # figure 5 in MSB
    pass
    return pl