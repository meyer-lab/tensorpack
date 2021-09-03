import numpy as np

class Tensor():
    def __init__(self, A: np.ndarray):
        self.A = A
        self.shape = A.shape
        self.order = len(self.shape)

        pass
    pass

