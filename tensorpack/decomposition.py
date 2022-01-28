import pickle
import numpy as np
from numpy.linalg import norm
from tensorly import partial_svd
from .cmtf import perform_CP, calcR2X
from .SVD_impute import IterativeSVD

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
        U, S, V = partial_svd(imp, min(dat.shape) - 1)
        scores = U @ np.diag(S)
        loadings = V
        recon = scores @ loadings
        new_diff = norm(imp[miss_idx] - recon[miss_idx]) / norm(recon[miss_idx])
        diff = new_diff
        imp[miss_idx] = recon[miss_idx]
    return imp


class Decomposition():
    def __init__(self, data, max_rr=5):
        self.data = data
        self.method = perform_CP
        self.rrs = np.arange(1,max_rr+1)
        pass

    def perform_tfac(self):
        self.tfac = [self.method(self.data, r=rr) for rr in self.rrs]
        self.tR2X = [c.R2X for c in self.tfac]
        self.tsize = [rr * sum(self.tfac[0].shape) for rr in self.rrs]

    def perform_PCA(self, flattenon=0):
        dataShape = self.data.shape
        flatData = np.reshape(np.moveaxis(self.data, flattenon, 0), (dataShape[flattenon], -1))
        if not np.all(np.isfinite(flatData)):
            flatData = impute_missing_mat(flatData)

        U, S, V = partial_svd(flatData, max(self.rrs))
        scores = U @ np.diag(S)
        loadings = V
        recon = [scores[:, :rr] @ loadings[:rr, :] for rr in self.rrs]
        self.PCAR2X = [calcR2X(c, mIn=flatData) for c in recon]
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

            for rr in self.rrs:
                tFac = self.method(missingCube, r=rr)
                Q2X[x,rr-1] = calcR2X(tFac, tIn=tImp)
                
        self.chordQ2X = Q2X

    def Q2X_entry(self, drop=10, repeat=5, comparePCA=True, flattenon=0):
        Q2X = np.zeros((repeat,self.rrs[-1]))
        # Q2X = np.zeros((np.sum(np.isfinite(self.data)), self.rrs[-1]))
        Q2XPCA = np.zeros((repeat,self.rrs[-1]))
        for x in range(repeat):
            missingCube = np.copy(self.data)
            tImp = np.copy(self.data)

            """
            # Option 1: checks if dropped values will create empty chords before removing
            for _ in range(drop):
                # drops entries
                removable = False
                while not removable:
                    idxs = np.argwhere(np.isfinite(missingCube))
                    i, j, k = idxs[np.random.choice(idxs.shape[0], 1)][0]
                    missingChordI = sum(np.isfinite(missingCube[:,j,k])) > 1
                    missingChordJ = sum(np.isfinite(missingCube[i,:,k])) > 1
                    missingChordK = sum(np.isfinite(missingCube[i,j,:])) > 1
                    if missingChordI and missingChordJ and missingChordK:
                        missingCube[i, j, k] = np.nan
                        removable = True 
            """
            """
            Option 2: find values that must be kept and remove from the remaining data
            1)  randomly find an minimum tensor to run PCA
            2)  prevent emin values from being removed in missingCube
            3)  remove # of values in missingCube
            4)  add emin values back in missingCube
            """

            chooseCube = np.isfinite(tImp)
            idxs = np.argwhere(chooseCube)

            modeidxs = np.zeros((tImp.ndim,max(tImp.shape)))
            # array representing the chords in every mode (row)
            for i in range(tImp.ndim):
                modeidxs[i] = [1 for n in range(tImp.shape[i])] + [0 for m in range(len(modeidxs[i])-tImp.shape[i])]
            # self.modeidxs = modeidxs
            
            counter = 0
            while np.sum(modeidxs) > 0:
                # number of columns with no values inside
                ranidx = np.random.choice(idxs.shape[0], 1) 
                i,j,k = idxs[ranidx][0]
                if modeidxs[0,i] > 0 or modeidxs[1,j] > 0 or modeidxs[2,k] > 0:
                    if modeidxs[0,i] > 0:
                        modeidxs[0,i] -= 1
                    if modeidxs[1,j] > 0:
                        modeidxs[1,j] -= 1
                    if modeidxs[2,k] > 0:
                        modeidxs[2,k] -= 1
                    np.delete(idxs, ranidx, axis=0)
                    counter += 1
            
            assert idxs.shape[0] >= drop
            # testing       print(str(counter) + " values withheld from drop")
            
            for _ in range(drop):
                i, j, k = idxs[np.random.choice(idxs.shape[0], 1)][0]
                missingCube[i,j,k] = np.nan

            # print(np.sum(np.isnan(missingCube)) + " values dropped") # testing
            
            for rr in self.rrs:
                tFac = self.method(missingCube, r=rr)
                Q2X[x,rr-1] = calcR2X(tFac, tIn=tImp)

            """
            # modified Q2X for testing single drops
            count = 0
            for id in idxs:
                missingCube[id[0],id[1],id[2]] = np.nan
                tImp[np.isfinite(missingCube)] = np.nan
                for rr in self.rrs:
                    tFac = self.method(missingCube, r=rr)
                    Q2X[count,rr-1] = calcR2X(tFac, tIn=tImp)
                count += 1
            """
            
            if comparePCA:
                si = IterativeSVD(rank=rank, random_state=1)
                missingMat = np.reshape(np.moveaxis(missingCube, flattenon, 0), (missingCube.shape[flattenon], -1))
                missingMat = impute_missing_mat(missingMat)
                mImp = np.reshape(np.moveaxis(tImp, flattenon, 0), (tImp.shape[flattenon], -1))

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