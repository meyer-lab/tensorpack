import numpy as np
import pandas as pd
from .cmtf import perform_CP, calcR2X
from statsmodels.multivariate.pca import PCA

class Decomposition():
    def __init__(self, data, max_rr=6):
        self.data = data
        self.method = perform_CP
        self.rrs = np.arange(1, max_rr)
        pass

    def perform_tfac(self):
        self.tfac = [self.method(self.data, r=rr) for rr in self.rrs]
        self.TR2X = [c.R2X for c in self.tfac]
        self.sizeT = [rr * sum(self.tfac[0].shape) for rr in self.rrs]

    def perform_PCA(self, flattenon=0):
        dataShape = self.data.shape
        flatData = np.reshape(np.moveaxis(self.data, flattenon, 0), (dataShape[flattenon], -1))
        self.PCA = PCA(flatData, ncomp=max(self.rrs), missing='fill-em', standardize=False, demean=False, normalize=False)
        recon = [self.PCA.scores[:, :rr] @ self.PCA.loadings.T[:rr, :] for rr in self.rrs]
        self.PCAR2X = [calcR2X(c, mIn = flatData) for c in recon]
        self.sizePCA = [sum(flatData.shape) * rr for rr in self.rrs]


    def Q2X_chord(self, drop=10, repeat=10):
        self.chordQ2X = None # df
        pass

    def Q2X_entry(self, drop=10, repeat=10):
        self.entryQ2X = None  # df
        pass

    pass