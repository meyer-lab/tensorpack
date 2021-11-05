import seaborn as sns
import pandas as pd
import numpy as np
import pickle

"""
pickle file contains two arrays:
pca_rs = [x1, x2, ... ,xr]
tensor_rs = [y1, y2, ... ,yr]
"""

def plot_r2x(pickle_file):
    with open(pickle_file, 'rb') as f:
        pickle.dump(pca_rs, f)
        pickle.dump(tensor_rs, f)
    r2x_list = [pca_rs[r].R2X for r in pca_rs, tensor_rs[r].R2X for r in tensor_rs]
    r2x_data = pd.DataFrame({'Number of Components': list(range(1, len(pca_rs)+1)) + list(range(1, len(tensor_rs)+1)),
                               'R2X': r2x_list,
                               'Method': ['PCA'] * len(pca_rs) + ['Tensor'] * len(tensor_rs)})
    pl = sns.scatterplot(data=r2x_data, x='Number of Components', y='R2X', hue='Method')
    pl.set(ylim=(0.0, 1.0))
    return pl

def plot_reduction(pickle_file):
    # figure 2b in MSB
    # Enio

    file = open(pickle_file, 'rb')
    object_file = pickle.load(pickle_file)
    pca_rs, tensor_rs = object_file

    tFacSize = list(tensor_rs.shape[0]*i for i in range(1,tensor_rs.shape[1]+1))
    PCSize = list(pca_rs.shape[0]*i for i in range(1,pca_rs.shape[1]+1))



    pass
    return pl

def plot_q2x_chord(pickle_file):
    # figure 3a in MSB
    pass
    return pl

def plot_q2x_entries(pickle_file):
    # figure 3b in MSB
    pass
    return pl

def plot_weights(pickle_file):
    # figure 5 in MSB
    pass
    return pl