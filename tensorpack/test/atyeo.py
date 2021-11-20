from os.path import join, dirname
import numpy as np
import pandas as pd
from ..cmtf import perform_CP
from ..figureCommon import subplotLabel, getSetup
from ..plot import *
from ..decomposition import Decomposition

path_here = dirname(dirname(__file__))

def load_file(name):
    """ Return a requested data file. """
    data = pd.read_csv(join(path_here, "test/" + name + ".csv"), delimiter=",", comment="#")

    return data


def getAxes():
    """ Get each of the axes over which the data is measured. """
    df = load_file("atyeo_covid")
    df = df.filter(regex='SampleID|Ig|Fc|SNA|RCA', axis=1)

    axes = df.filter(regex='Ig|Fc|SNA|RCA', axis=1)
    axes = axes.columns.str.split(" ", expand = True)

    subject = df['SampleID']
    subject = subject[0:22]

    antigen = []
    receptor = []

    for row in axes:
        if (row[0] not in antigen):
            antigen.append(row[0])
        if (row[1] not in receptor):
            receptor.append(row[1])

    return subject, receptor, antigen
    

def createCube():
    """ Import the data and assemble the antigen cube. """
    subject, receptor, antigen = getAxes()
    cube = np.full([len(subject), len(receptor), len(antigen)], np.nan)
    
    df = load_file("atyeo_covid")
    df = df.filter(regex='Ig|Fc|SNA|RCA', axis=1)
    df = df[0:len(subject)]

    for i, row in df.iterrows():
        for j in range(len(receptor)):
            rec =  df.filter(regex=receptor[j])
            cube[i,j] = rec.iloc[i,:]
    
    # Check that there are no slices with completely missing data        
    assert ~np.any(np.all(np.isnan(cube), axis=(0, 1)))
    assert ~np.any(np.all(np.isnan(cube), axis=(0, 2)))
    assert ~np.any(np.all(np.isnan(cube), axis=(1, 2)))

    return cube 

def sample_figure():
    ax, f = getSetup((8, 5), (2, 3))
    atyeo = Decomposition(createCube())
    atyeo.perform_decomp()
    atyeo.perform_PCA()
    tfacr2x(ax[0], atyeo)
    reduction(ax[1], atyeo)
    return f