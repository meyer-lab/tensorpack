"""
Canonical Polyadic
"""

import numpy as np
import tensorly as tl
from tensorly.tenalg import khatri_rao
from copy import deepcopy
from tensorly.decomposition._cp import initialize_cp, parafac
from fancyimpute import SoftImpute
from .cmtf import calcR2X, censored_lstsq, reorient_factors, sort_factors


