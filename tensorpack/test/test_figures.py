from .atyeo import createCube
from ..figures import *
from ..decomposition import Decomposition


ax, f = getSetup((8, 6), (2, 3))
atyeo = Decomposition(createCube())
atyeo.perform_decomp()


from tensorpack.figures import *
from tensorpack.figureCommon import getSetup
from tensorpack.test.atyeo import createCube
from tensorpack.decomposition import *