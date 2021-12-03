import pickle
import numpy as np
import pandas as pd
from pandas._libs import missing
from pandas.core.frame import DataFrame
from statsmodels.multivariate.pca import PCA
from sklearn.decomposition import TruncatedSVD
from tensorpack.cmtf import perform_CP, calcR2X


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

        tsvd = TruncatedSVD(n_components=max(self.rrs))
        scores = tsvd.fit_transform(flatData)
        loadings = tsvd.components_
        recon = [scores[:, :rr] @ loadings[:rr, :] for rr in self.rrs]
        self.PCAR2X = [calcR2X(c, mIn = flatData) for c in recon]
        self.sizePCA = [sum(flatData.shape) * rr for rr in self.rrs]


    def Q2X_chord(self, drop=2, repeat=2):
        Q2X = np.zeros((repeat,self.rrs[-1]))
        for x in range(repeat):
            missingCube = np.copy(self.data)
            for _ in range(drop):
                idxs = np.argwhere(np.isfinite(missingCube))
                i, j, k = idxs[np.random.choice(idxs.shape[0], 1)][0]
                missingCube[:, j, k] = np.nan
            tenFacs = [self.method(missingCube, r=rr) for rr in self.rrs]
            tImps = [c.to_tensor() for c in tenFacs]
            tIn = np.copy(self.data)
            tIn[np.isfinite(missingCube)] = np.nan
            tMask = np.isfinite(tIn)
            for c,tImp in enumerate(tImps):
                Top = np.sum(np.square(tImp * tMask - np.nan_to_num(tIn)))
                Bottom = np.sum(np.square(np.nan_to_num(tIn)))
                Q2X[x,c] = 1 - Top/Bottom
                
        self.chordQ2X = Q2X # df

    def Q2X_entry(self, drop=10, repeat=10):
        self.entryQ2X = None  # df
        pass

    def save(self, pfile):
        with open(pfile, "wb") as output_file:
            pickle.dump(self.__dict__, output_file)

    def load(self, pfile):
        with open(pfile, "rb") as input_file:
            tmp_dict = pickle.load(input_file)
            self.__dict__.update(tmp_dict)

    pass