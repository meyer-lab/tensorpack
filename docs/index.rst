tensorpack
==========

tensorpack is a collection of tensor decomposition methods from the Meyer lab,
built on TensorLy and xarray. It targets multi-dimensional biological data sets
that are often incomplete or come paired with additional matrices.

Installation
------------

.. code-block:: bash

   uv add git+https://github.com/meyer-lab/tensorpack.git@main

Which method should I use?
--------------------------

- :func:`tensorpack.cmtf.perform_CP` -- CP decomposition of one tensor; ``NaN``
  entries are treated as missing.
- :func:`tensorpack.cmtf.perform_CMTF` -- a tensor and a matrix that share their
  first mode, factored jointly.
- :class:`tensorpack.coupled.CoupledTensor` -- any number of arrays sharing named
  dimensions in an :class:`xarray.Dataset`, optionally non-negative.
- :class:`tensorpack.decomposition.Decomposition` -- scan component counts and
  compute R2X and Q2X.

Example
-------

.. code-block:: python

   import numpy as np
   from tensorpack import perform_CP

   rng = np.random.default_rng(0)
   tensor = rng.gamma(2, 2, (10, 20, 25))
   tensor[rng.random(tensor.shape) < 0.2] = np.nan

   cp = perform_CP(tensor, r=3)
   print(cp.R2X)

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   api
