import pickle
import numpy as np
from .cmtf import perform_CP, calcR2X
from tensorly import partial_svd
from .SVD_impute import IterativeSVD


def create_missingness(tensor, drop):
    """
    Creates missingness for a full tensor.
    """
    idxs = np.argwhere(np.isfinite(tensor))
    dropidxs = idxs[np.random.choice(idxs.shape[0], drop, replace=False)]
    dropidxs = tuple(dropidxs.T)
    tensor[dropidxs] = np.nan


def entry_drop(tensor, drop):
    """
    Drops random values within a tensor. Finds a bare minimum cube before dropping values to ensure PCA remains viable.

    Parameters
    ----------
    tensor : ndarray
        Takes a tensor of any shape. Preference for at least two values present per chord.
    drop : int
        To set a percentage, multiply np.sum(np.isfinite(tensor)) by the percentage
        to find the relevant drop value, rounding to nearest int.

    Returns
    -------
    None : tensor is modified with missing values.
    """
    # Track chords for each mode to ensure bare minimum cube covers each chord at least once
    midxs = np.zeros((tensor.ndim,max(tensor.shape)))
    for i in range(tensor.ndim):
        midxs[i] = [1 for n in range(tensor.shape[i])] + [0 for m in range(len(midxs[i])-tensor.shape[i])]
    modecounter = np.arange(tensor.ndim)

    # Remove bare minimum cube idxs from droppable values
    idxs = np.argwhere(np.isfinite(tensor))
    while np.sum(midxs) > 0:
        removable = False
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
    tensor[dropidxs] = np.nan


def chord_drop(tensor, drop):
    """
    Removes chords along axis = 0 of a tensor.

    Parameters
    ----------
    tensor : ndarray
        Takes a tensor of any shape.
    drop : int
        To set a percentage, multiply np.sum(tensor.ndim[0]) by the percentage 
        to find the relevant drop value, rounding to nearest int.

    Returns
    -------
    None : tensor is modified with missing chords.
    """
    # Drop chords based on random idxs
    chordlen = tensor.shape[0]
    for _ in range(drop):
        idxs = np.argwhere(np.isfinite(tensor))
        chordidx = np.delete(idxs[np.random.choice(idxs.shape[0], 1)][0],0,-1)
        dropidxs = []
        for i in range(chordlen):
            dropidxs.append(tuple(np.insert(chordidx,0,i).T))
        for i in range(chordlen):
            tensor[dropidxs[i]] = np.nan


class Decomposition():
    def __init__(self, data, max_rr=5, method=perform_CP):
        """
        Decomposition object designed for plotting.

        Parameters
        ----------
        data : ndarray
            Takes a tensor of any shape.
        max_rr : int
            Defines the maximum component to consider during factorization.
        method : function
            Takes a factorization method. Default set to perform_CP() from cmtf.py
        """
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

    def Q2X_chord(self, drop=5, repeat=3, mode=0):
        """
        Calculates Q2X when dropping chords along axis = mode from the data using self.method for factor decomposition,
        comparing each component. Drops in Q2X from one component to the next may signify overfitting.

        Parameters
        ----------
        drop : int
            To set a percentage, use np.sum(tensor.ndim[mode]) and multiply by the percentage 
            to find the relevant drop value, rounding to nearest int.
        repeat : int
        mode : int
            Defaults to mode corresponding to axis = 0. Can be set to any mode of the tensor.

        Returns
        -------
        self.Q2X : ndarray of size (repeat, max_rr)
            Each value in a row represents the Q2X of the tensor calculated for components 1 to max_rr.
            Each row represents a single repetition.
        """
        Q2X = np.zeros((repeat,self.rrs[-1]))

        for x in range(repeat):
            missingCube = np.copy(self.data)
            np.moveaxis(missingCube,mode,0)
            tImp = np.copy(self.data)
            np.moveaxis(tImp,mode,0)
            chord_drop(missingCube, drop)

            # Calculate Q2X for each number of components
            tImp[np.isfinite(missingCube)] = np.nan
            for rr in self.rrs:
                tFac = self.method(missingCube, r=rr)
                Q2X[x,rr-1] = calcR2X(tFac, tIn=tImp)

        self.chordQ2X = Q2X

    def Q2X_entry(self, drop=20, repeat=3, comparePCA=True):
        """
        Calculates Q2X when dropping entries from the data using self.method for factor decomposition,
        comparing each component. Drops in Q2X from one component to the next may signify overfitting.

        Parameters
        ----------
        drop : int
            To set a percentage, multiply np.sum(np.isfinite(tensor)) by the percentage 
            to find the relevant drop value, rounding to nearest int.
        repeat : int
        comparePCA : boolean
            Defaulted to calculate Q2X for respective principal components using PCA for factorization
            to compare against self.method.

        Returns
        -------
        self.Q2X : ndarray of size (repeat, max_rr)
            Each value in a row represents the Q2X of the tensor calculated for components 1 to max_rr using self.method.
            Each row represents a single repetition.
        self.Q2XPCA : ndarray of size (repeat, max_rr)
            Each value in a row represents the Q2X of the tensor calculated for components 1 to max_rr using PCA after
            SVD imputation. Each row represents a single repetition.
        """
        Q2X = np.zeros((repeat,self.rrs[-1]))
        Q2XPCA = np.zeros((repeat,self.rrs[-1]))
        
        for x in range(repeat):
            missingCube = np.copy(self.data)
            tImp = np.copy(self.data)
            entry_drop(missingCube, drop)

            # Calculate Q2X for each number of components
            tImp[np.isfinite(missingCube)] = np.nan
            for rr in self.rrs:
                tFac = self.method(missingCube, r=rr)
                Q2X[x,rr-1] = calcR2X(tFac, tIn=tImp)
            
            # Calculate Q2X for each number of principal components using PCA for factorization as comparison
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
