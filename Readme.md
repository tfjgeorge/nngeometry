# NNGeometry

[![Build Status](https://travis-ci.org/tfjgeorge/nngeometry.svg?branch=master)](https://travis-ci.org/tfjgeorge/nngeometry) [![codecov](https://codecov.io/gh/tfjgeorge/nngeometry/branch/master/graph/badge.svg)](https://codecov.io/gh/tfjgeorge/nngeometry) [![DOI](https://zenodo.org/badge/208082966.svg)](https://zenodo.org/badge/latestdoi/208082966) [![PyPI version](https://badge.fury.io/py/nngeometry.svg)](https://badge.fury.io/py/nngeometry)



NNGeometry allows you to:
 - compute **Fisher Information Matrices** (FIM) or derivates, using efficient approximations such as low-rank matrices, KFAC, diagonal and so on.
 - compute finite **Neural Tangent Kernels**, even for multiple output functions.
 - easily and efficiently compute linear algebra operations involving these matrices **regardless of their approximation**.

## Example

In the Elastic Weight Consolidation continual learning technique, you want to compute <img src="https://render.githubusercontent.com/render/math?math=\left(\mathbf{w}-\mathbf{w}_{A}\right)^{\top}F\left(\mathbf{w}-\mathbf{w}_{A}\right)">. It can be achieved with a diagonal approximation for the FIM using: 
```python
F = FIM(model=model,
        loader=loader,
        representation=PMatDiag,
        n_output=10)

regularizer = F.vTMv(w - w_a)
```
If diagonal is not sufficiently accurate then you could instead choose a KFAC approximation, by just changing `PMatDiag` to `PMatKFAC` in the above.

## Documentation

For more examples, you can visit the documentation at https://nngeometry.readthedocs.io

## Citation

If you use NNGeometry in a published project, please cite our work using the following bibtex entry

```tex
@software{george_nngeometry,
  author       = {Thomas George},
  title        = {{NNGeometry: Easy and Fast Fisher Information 
                   Matrices and Neural Tangent Kernels in PyTorch}},
  month        = feb,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {v0.2.1},
  doi          = {10.5281/zenodo.4532597},
  url          = {https://doi.org/10.5281/zenodo.4532597}
}
```
