"""Partial least squares (PLS) for tensors, including coupled (CMTF) variants."""

from .cmtf import ctPLS
from .tpls import tPLS

__all__ = ["ctPLS", "tPLS"]
