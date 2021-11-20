from .atyeo import createCube
from ..plot import *
from ..decomposition import Decomposition



atyeo = Decomposition(createCube())
atyeo.perform_decomp()


from tensorpack.plot import *
from tensorpack.figureCommon import getSetup
from tensorpack.test.atyeo import createCube
from tensorpack.decomposition import *