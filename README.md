# tensorpack

A collection of tensor decomposition methods from the [Meyer lab](https://meyerlab.org), built on [TensorLy](https://tensorly.org) and [xarray](https://xarray.dev). It is aimed at biological data sets that are naturally multi-dimensional (for example subjects × antigens × measurements) and are often incomplete or paired with additional matrices.

## Features

- **CP decomposition with missing data** (`perform_CP`): CANDECOMP/PARAFAC fit by alternating least squares that ignores `NaN` entries instead of requiring imputation.
- **Coupled matrix–tensor factorization** (`perform_CMTF`): jointly factors a tensor and a matrix that share their first mode.
- **Coupled tensor factorization of xarray datasets** (`CoupledTensor`): factors any number of arrays that share named dimensions, with optional non-negativity.
- **Tucker decomposition** (`tucker_decomp`) and a `Decomposition` helper that scans component counts and reports variance explained (R2X) and cross-validated, missing-value-based prediction accuracy (Q2X).
- **Imputation and masking utilities**: an iterative-SVD imputer (`IterativeSVD`) and functions to hold out entries or chords for cross-validation (`tensorpack.impute`).
- **Plotting helpers** for R2X curves and factor heatmaps (`tensorpack.plot`, `tensorpack.xplots`).

## Installation

tensorpack requires Python 3.11 or 3.12. To use it in another project, add it from GitHub:

```bash
uv add git+https://github.com/meyer-lab/tensorpack.git@main
# or, with pip
pip install git+https://github.com/meyer-lab/tensorpack.git@main
```

## Quick start

### CP decomposition of a tensor with missing values

```python
import numpy as np
from tensorpack import perform_CP

rng = np.random.default_rng(0)
tensor = rng.gamma(2, 2, (10, 20, 25))
tensor[rng.random(tensor.shape) < 0.2] = np.nan  # 20% missing

cp = perform_CP(tensor, r=3)
cp.R2X  # fraction of variance explained
cp.factors  # one (size × 3) factor matrix per mode
cp.weights  # per-component weights
```

### Coupled matrix–tensor factorization

```python
from tensorpack import perform_CMTF

matrix = rng.gamma(2, 2, (10, 15))  # shares its first mode with `tensor`
cmtf = perform_CMTF(tensor, matrix, r=3)
cmtf.factors  # tensor factors; factors[0] is shared with the matrix
cmtf.mFactor  # factor for the matrix's second mode
```

### Coupled factorization of several xarray arrays

```python
import xarray as xr
from tensorpack import CoupledTensor

data = xr.Dataset(
    {
        "x": (["a", "b", "c"], rng.random((8, 6, 5))),
        "y": (["a", "d"], rng.random((8, 4))),
    }
)  # dimension "a" is shared

ct = CoupledTensor(data, rank=3)
ct.initialize("svd")  # or "nmf" (with fit(nonneg=True)) / "randomized_svd"
ct.fit()
ct.R2X(), ct.R2X("x")  # overall and per-array variance explained
```

## Development

The project is managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync          # create the environment
make test        # run the test suite
make coverage.xml  # tests with coverage
```

## License

MIT. See [LICENSE](LICENSE).
