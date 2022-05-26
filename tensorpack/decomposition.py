import pickle
from re import A
import numpy as np
from .cmtf import perform_CMTF, perform_CP, calcR2X
from tensorly import partial_svd
from .SVD_impute import IterativeSVD
from .tucker import tucker_decomp
from .impute import create_missingness, entry_drop, joint_entry_drop, chord_drop

class Decomposition():
    def __init__(self, data, matrix=[0], max_rr=5, method=perform_CP):
        """
        Decomposition object designed for plotting. Capable of handling a single tensor and matrix jointly.

        Parameters
        ----------
        data : ndarray
            Takes a tensor of any shape.
        matrix : ndarray (optional)
            Takes a matrix of any shape.
        max_rr : int
            Defines the maximum component to consider during factorization.
        method : function
            Takes a factorization method. Default set to perform_CP() from cmtf.py
            other methods include: tucker_decomp
        """
        self.data = data
        self.method = method
        self.rrs = np.arange(1,max_rr+1)
        self.hasMatrix = False
        if isinstance(matrix, np.ndarray):
            if matrix.ndim == 2:
                self.matrix = matrix
                self.hasMatrix = True
        pass

    def perform_tfac(self, callback=None, rr_callback=1):
        if callback:
            self.method(self.data, r=rr_callback, callback=callback)
        self.tfac = [self.method(self.data, r=rr) for rr in self.rrs]
        self.TR2X = [calcR2X(c, tIn=self.data) for c in self.tfac]
        self.sizeT = [rr * sum(self.tfac[0].shape) for rr in self.rrs]

    def perform_tucker(self):
        """ Try out Tucker for up to a specific number of ranks. """
        self.Tucker, self.TuckErr, self.TuckRank = self.method(self.data, max(self.rrs)+1)

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

    def Q2X_chord(self, drop=5, repeat=3, mode=0, callback=None, rr_callback=1):
        """
        Calculates Q2X when dropping chords along axis = mode from the data using self.method for factor decomposition,
        comparing each component. Drops in Q2X from one component to the next may signify overfitting.

        Parameters
        ----------
        drop : int
            To set a percentage, tensor.shape[mode] and multiply by the percentage 
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
                if rr == rr_callback and callback:
                    tFac = self.method(missingCube, r=rr, callback=callback)
                else:  
                    tFac = self.method(missingCube, r=rr)
                Q2X[x,rr-1] = calcR2X(tFac, tIn=tImp)

        self.chordQ2X = Q2X

    def Q2X_entry(self, drop=20, repeat=3, comparePCA=True, callback=None, rr_callback=1):
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
                if rr == rr_callback and callback:
                    tFac = self.method(missingCube, r=rr, callback=callback)
                else:  
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
