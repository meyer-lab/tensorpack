"""
Testing Decomposition
"""

from ..decomposition import Decomposition
from .atyeo import createCube
import os

def test_decomp_obj():
    a = Decomposition(createCube())
    a.perform_tfac()
    a.perform_PCA()
    assert len(a.PCAR2X) == len(a.sizePCA)
    assert len(a.TR2X) == len(a.sizeT)

    # test decomp save
    fname = "test_temp_atyeo.pkl"
    a.save(fname)
    b = Decomposition(None)
    b.load(fname)
    assert len(b.PCAR2X) == len(b.sizePCA)
    assert len(b.TR2X) == len(b.sizeT)
    os.remove(fname)
