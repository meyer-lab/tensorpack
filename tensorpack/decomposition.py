import numpy as np
import pandas as pd
from .cmtf import perform_CP

class decomposition():
    def __init__(self, data):
        self.data = data
        self.method = perform_CP
        self.comps = np.arange(1, 6)
        pass

    def perform_decomp(self):
        self.tfac = [self.method(self.data, r=rr) for rr in self.comps]
        self.TR2X = [c.R2X for c in self.tfac]

    def perform_PCA(self):
        ## insert PCA here
        self.PCAR2X = []
        pass

    def Q2X_chord(self, drop=10, repeat=10):
        self.chordQ2X = None # df
        pass

    def Q2X_entry(self, drop=10, repeat=10):
        pass

    pass