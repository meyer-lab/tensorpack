"""Pytest configuration: pin BLAS to one thread before numpy is imported.

The test problems are tiny, so multithreaded BLAS spends far more time on
synchronization than on arithmetic (and oversubscribes cores under xdist).
"""

import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _seed_global_rng():
    """Seed NumPy's global RNG so tests on random data are reproducible."""
    np.random.seed(0)
