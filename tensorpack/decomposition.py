import pickle
import numpy as np
from .cmtf import perform_CP, calcR2X
from tensorly import partial_svd
from .SVD_impute import IterativeSVD


class Decomposition():
    def __init__(self, data, max_rr=5, method=perform_CP):
        self.data = data
        self.method = method
        self.rrs = np.arange(1,max_rr+1)
        pass

    def perform_tfac(self):
        self.tfac = [self.method(self.data, r=rr) for rr in self.rrs]
        self.TR2X = [calcR2X(c, tIn=self.data) for c in self.tfac]
        self.sizeT = [rr * sum(self.tfac[0].shape) for rr in self.rrs]

    def perform_PCA(self, flattenon=0):
        dataShape = self.data.shape
        flatData = np.reshape(np.moveaxis(self.data, flattenon, 0), (dataShape[flattenon], -1))
        if not np.all(np.isfinite(flatData)):
            flatData = IterativeSVD(rank=1, random_state=1).fit_transform(flatData)

        U, S, V = partial_svd(flatData, max(self.rrs))
        scores = U @ np.diag(S)
        loadings = V
        recon = [scores[:, :rr] @ loadings[:rr, :] for rr in self.rrs]
        self.PCAR2X = [calcR2X(c, mIn=flatData) for c in recon]
        self.sizePCA = [sum(flatData.shape) * rr for rr in self.rrs]

    def Q2X_chord(self, drop=5, repeat=5, mode=0):
        Q2X = np.zeros((repeat,self.rrs[-1]))
        for x in range(repeat):
            missingCube = np.copy(self.data)
            np.moveaxis(missingCube,mode,0)
            tImp = np.copy(self.data)
            np.moveaxis(tImp,mode,0)
            chordlen = missingCube.shape[0]
            for _ in range(drop):
                idxs = np.argwhere(np.isfinite(tImp))
                chordidx = np.delete(idxs[np.random.choice(idxs.shape[0], 1)][0],0,-1)
                dropidxs = []
                for i in range(chordlen):
                    dropidxs.append(tuple(np.insert(chordidx,0,i).T))
                for i in range(chordlen):
                    missingCube[dropidxs[i]] = np.nan

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
            midxs = np.zeros((tImp.ndim,max(tImp.shape)))
            for i in range(tImp.ndim):
                midxs[i] = [1 for n in range(tImp.shape[i])] + [0 for m in range(len(midxs[i])-tImp.shape[i])]
            modecounter = np.arange(tImp.ndim)
            while np.sum(midxs) > 0:
                removable=False
                ran = np.random.choice(idxs.shape[0], 1) 
                ranidx = idxs[ran][0]
                counter = 0
                for i in ranidx:
                    if midxs[modecounter[counter],i] > 0:
                        removable = True
                    midxs[modecounter[counter],i] = 0
                    counter += 1
                if removable == True:
                    idxs = np.delete(idxs, ran, axis=0)
            assert idxs.shape[0] >= drop

            dropidxs = idxs[np.random.choice(idxs.shape[0], drop, replace=False)]
            dropidxs = tuple(dropidxs.T)
            missingCube[dropidxs] = np.nan

            tImp[np.isfinite(missingCube)] = np.nan
            for rr in self.rrs:
                tFac = self.method(missingCube, r=rr)
                Q2X[x,rr-1] = calcR2X(tFac, tIn=tImp)
            
            if comparePCA:
                si = IterativeSVD(rank=max(self.rrs), random_state=1)
                missingMat = np.reshape(np.moveaxis(missingCube, 0, 0), (missingCube.shape[0], -1))
                mImp = np.reshape(np.moveaxis(tImp, 0, 0), (tImp.shape[0], -1))

                missingMat = si.fit_transform(missingMat)
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
