# tensorpack
A collection of tensor methods from the Meyer lab.

To add it to your Python package, add the following line to `requirements.txt` and remake `venv`:
```
git+https://github.com/meyer-lab/tensorpack.git@main
```

## Partial least squares

`tensorpack.pls` contains tensor partial least squares (`tPLS`) and its coupled
matrix-tensor factorization variant (`ctPLS`), moved over from the former
`cmtf-pls` repository.

```python
from tensorpack.pls import tPLS, ctPLS
```
