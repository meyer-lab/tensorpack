"""Pytest configuration: pin BLAS to one thread before numpy is imported.

The test problems are tiny, so multithreaded BLAS spends far more time on
synchronization than on arithmetic (and oversubscribes cores under xdist).
"""

import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")
