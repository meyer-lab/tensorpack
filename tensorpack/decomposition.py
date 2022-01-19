import pickle
import numpy as np
import pandas as pd
from numpy.linalg import norm
from sklearn.decomposition import TruncatedSVD
from .cmtf import perform_CP, calcR2X

def flatten_to_mat(tensor):
    n = tensor.shape[0]
    tflat = np.reshape(tensor, (n, -1))
    if not np.all(np.isfinite(tflat)):
        tflat = impute_missing_mat(tflat)
    return tflat

def impute_missing_mat(dat):
    miss_idx = np.where(~np.isfinite(dat))
    if len(miss_idx[0]) <= 0:
        return dat
    assert np.all(np.any(np.isfinite(dat), axis=0)), "Cannot impute if an entire column is empty"
    assert np.all(np.any(np.isfinite(dat), axis=1)), "Cannot impute if an entire row is empty"

    imp = np.copy(dat)
    col_mean = np.nanmean(dat, axis=0, keepdims=True)
    imp[miss_idx] = np.take(col_mean, miss_idx[1])

    diff = 1.0
    while diff > 1e-3:
        tsvd = TruncatedSVD(n_components=min(dat.shape)-1)
        scores = tsvd.fit_transform(imp)
        loadings = tsvd.components_
        recon = scores @ loadings
        new_diff = norm(imp[miss_idx] - recon[miss_idx]) / norm(recon[miss_idx])
        assert new_diff < diff, "Matrix imputation difference is not decreasing"
        diff = new_diff
        imp[miss_idx] = recon[miss_idx]
    return imp

class Decomposition():
    def __init__(self, data, max_rr=6):
        self.data = data
        self.method = perform_CP
        self.rrs = np.arange(1, max_rr)
        pass

    def perform_tfac(self):
        self.tfac = [self.method(self.data, r=rr) for rr in self.rrs]
        self.tR2X = [c.R2X for c in self.tfac]
        self.tsize = [rr * sum(self.tfac[0].shape) for rr in self.rrs]

    def perform_PCA(self):
        flatData = flatten_to_mat(self.data)
        tsvd = TruncatedSVD(n_components=max(self.rrs))
        scores = tsvd.fit_transform(flatData)
        loadings = tsvd.components_
        recon = [scores[:, :rr] @ loadings[:rr, :] for rr in self.rrs]
        self.PCAR2X = [calcR2X(c, mIn = flatData) for c in recon]
        self.sizePCA = [sum(flatData.shape) * rr for rr in self.rrs]


    def Q2X_chord(self, drop=10, repeat=10):
        Q2X = np.zeros(repeat,self.rrs[-1])
        for x in range(repeat):
            missingCube = np.copy(self.data)
            for _ in range(drop):
                # drops chords
                idxs = np.argwhere(np.isfinite(missingCube))
                i, j, k = idxs[np.random.choice(idxs.shape[0], 1)][0]
                missingCube[:, j, k] = np.nan
            
            tImp = np.copy(self.data)
            tImp[np.isfinite(missingCube)] = np.nan

            for rr in enumerate(self.rrs):
                tFac = self.method(missingCube, rr)
                Q2X[x,rr] = calcR2X(tFac, tIn=tImp)
                
        self.chordQ2X = Q2X

    def Q2X_entry(self, drop=10, repeat=10, comparePCA=True, flattenon=0):
        Q2X = np.zeros((repeat,self.rrs[-1]))
        Q2XPCA = np.zeros((repeat,self.rrs[-1]))
        for x in range(repeat):
            missingCube = np.copy(self.data)
            for _ in range(drop):
                # drops entries
                removable = False
                attempt = 0
                while not removable:
                    # checks if dropped values will create empty chords before removing
                    if attempt == 20:
                        break
                        # maximum 20 attempts; tensor may be too sparse
                    idxs = np.argwhere(np.isfinite(missingCube))
                    i, j, k = idxs[np.random.choice(idxs.shape[0], 1)][0]
                    missingChordI = sum(np.isfinite(missingCube[:,j,k])) > 1
                    missingChordJ = sum(np.isfinite(missingCube[i,:,k])) > 1
                    missingChordK = sum(np.isfinite(missingCube[i,j,:])) > 1
                    if missingChordI and missingChordJ and missingChordK:
                        missingCube[i, j, k] = np.nan
                        removable = True
                    attempt += 1     

            tImp = np.copy(self.data)
            tImp[np.isfinite(missingCube)] = np.nan

            for rr in enumerate(self.rrs):
                tFac = self.method(missingCube, rr)
                Q2X[x,rr] = calcR2X(tFac, tIn=tImp)

            if comparePCA:
                missingMat = flatten_to_mat(missingCube)
                mImp = flatten_to_mat(self.data)
                mImp[np.isfinite(missingCube)] = np.nan

                tsvd = TruncatedSVD(n_components=max(self.rrs))
                scores = tsvd.fit_transform(missingMat)
                loadings = tsvd.components_
                recon = [scores[:, :rr] @ loadings[:rr, :] for rr in self.rrs]
                Q2XPCA[x,rr] = [calcR2X(c, mIn = mImp) for c in recon]
    
            self.entryQ2X = Q2X
            self.entryQ2XPCA = Q2XPCA      

    def save(self, pfile):
        with open(pfile, "wb") as output_file:
            pickle.dump(self.__dict__, output_file)

    def load(self, pfile):
        with open(pfile, "rb") as input_file:
            tmp_dict = pickle.load(input_file)
            self.__dict__.update(tmp_dict)

    pass