import pickle
import numpy as np
from numpy.linalg import norm
from .cmtf import perform_CP, calcR2X
from tensorly import partial_svd
from .SVD_impute import IterativeSVD

def impute_missing_mat(dat):
    import warnings
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
        U, S, V = partial_svd(imp, min(dat.shape) - 1)
        scores = U @ np.diag(S)
        loadings = V
        recon = scores @ loadings
        new_diff = norm(imp[miss_idx] - recon[miss_idx]) / norm(recon[miss_idx])
        if new_diff > diff:
            warnings.warn("Matrix imputation difference is not decreasing", RuntimeWarning)
        diff = new_diff
        imp[miss_idx] = recon[miss_idx]
    return imp


class Decomposition():
    def __init__(self, data, max_rr=5, method=perform_CP):
        self.data = data
        self.method = method
        self.rrs = np.arange(1,max_rr+1)
        pass

    def perform_tfac(self):
        self.Tfac = [self.method(self.data, r=rr) for rr in self.rrs]
        self.TR2X = [calcR2X(c, tIn=self.data) for c in self.Tfac]
        self.Tsize = [rr * sum(self.Tfac[0].shape) for rr in self.rrs]

    def perform_PCA(self, flattenon=0):
        dataShape = self.data.shape
        flatData = np.reshape(np.moveaxis(self.data, flattenon, 0), (dataShape[flattenon], -1))
        if not np.all(np.isfinite(flatData)):
            si = IterativeSVD(rank=max(self.rrs), random_state=1)
            flatData = si.fit_transform(flatData)

        U, S, V = partial_svd(flatData, max(self.rrs))
        scores = U @ np.diag(S)
        loadings = V
        recon = [scores[:, :rr] @ loadings[:rr, :] for rr in self.rrs]
        self.PCAR2X = [calcR2X(c, mIn=flatData) for c in recon]
        self.sizePCA = [sum(flatData.shape) * rr for rr in self.rrs]

    def Q2X_chord(self, drop=5, repeat=5):
        Q2X = np.zeros((repeat,self.rrs[-1]))
        for x in range(repeat):
            missingCube = np.copy(self.data)
            tImp = np.copy(self.data)
            
            for _ in range(drop):
                removable = False
                while not removable:
                    idxs = np.argwhere(np.isfinite(missingCube))
                    i, j, k = idxs[np.random.choice(idxs.shape[0], 1)][0]
                    selection = np.random.choice([0,1,2])
                    if selection == 0:
                        if np.sum(np.isfinite(missingCube[:, j, k])) > 1:
                            missingCube[:, j, k] = np.nan
                            removable = True
                    elif selection == 1:
                        if np.sum(np.isfinite(missingCube[i, :, k])) > 1:
                            missingCube[i, :, k] = np.nan
                            removable = True
                    elif selection == 2:
                        if np.sum(np.isfinite(missingCube[i, j, :])) > 1:
                            missingCube[i, j, :] = np.nan
                            removable = True
            
            tImp[np.isfinite(missingCube)] = np.nan
            for rr in self.rrs:
                tFac = self.method(missingCube, r=rr)
                Q2X[x,rr-1] = calcR2X(tFac, tIn=tImp)
                
        self.chordQ2X = Q2X

    def Q2X_entry(self, drop=20, repeat=5, comparePCA=True):
        Q2X = np.zeros((repeat,self.rrs[-1]))
        Q2XPCA = np.zeros((repeat,self.rrs[-1]))
        for x in range(repeat):
            missingCube = np.copy(self.data)
            tImp = np.copy(self.data)
            idxs = np.argwhere(np.isfinite(tImp))
            
            # finds values that must be kept and only allow dropped values from the remaining data
            modeidxs = np.zeros((tImp.ndim,max(tImp.shape)))
            for i in range(tImp.ndim):
                modeidxs[i] = [1 for n in range(tImp.shape[i])] + [0 for m in range(len(modeidxs[i])-tImp.shape[i])]
            while np.sum(modeidxs) > 0:
                ranmidx = np.random.choice(idxs.shape[0], 1) 
                i,j,k = idxs[ranmidx][0]
                if modeidxs[0,i] > 0 or modeidxs[1,j] > 0 or modeidxs[2,k] > 0:
                    modeidxs[0,i] = 0
                    modeidxs[1,j] = 0
                    modeidxs[2,k] = 0
                    np.delete(idxs, ranmidx, axis=0)
            assert idxs.shape[0] >= drop
            # print(str(counter) + " values withheld from drop")
            ranidxs = np.random.choice(idxs.shape[0], drop, replace=False)
            for idx in ranidxs:
                i, j, k = idxs[idx]
                missingCube[i,j,k] = np.nan
            # print(str(np.sum(np.isnan(missingCube))) + " values dropped")

            tImp[np.isfinite(missingCube)] = np.nan
            for rr in self.rrs:
                tFac = self.method(missingCube, r=rr)
                Q2X[x,rr-1] = calcR2X(tFac, tIn=tImp)
            
            if comparePCA:
                si = IterativeSVD(rank=max(self.rrs), random_state=1)
                missingMat = np.reshape(np.moveaxis(missingCube, 0, 0), (missingCube.shape[0], -1))
                missingMat = si.fit_transform(missingMat)
                mImp = np.reshape(np.moveaxis(tImp, 0, 0), (tImp.shape[0], -1))

                U, S, V = partial_svd(missingMat, max(self.rrs))
                scores = U @ np.diag(S)
                loadings = V
                recon = [scores[:, :rr] @ loadings[:rr, :] for rr in self.rrs]
                Q2XPCA[x,:] = [calcR2X(c, mIn = mImp) for c in recon]
                self.entryQ2XPCA = Q2XPCA
    
        self.entryQ2X = Q2X
    

    def save(self, pfile):
        with open(pfile, "wb") as output_file:
            pickle.dump(self.__dict__, output_file)

    def load(self, pfile):
        with open(pfile, "rb") as input_file:
            tmp_dict = pickle.load(input_file)
            self.__dict__.update(tmp_dict)

    pass
