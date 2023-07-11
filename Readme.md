# NNGeometry

![Build Status](https://github.com/tfjgeorge/nngeometry/actions/workflows/nngeometry.yml/badge.svg) [![codecov](https://codecov.io/gh/tfjgeorge/nngeometry/branch/master/graph/badge.svg)](https://codecov.io/gh/tfjgeorge/nngeometry) [![DOI](https://zenodo.org/badge/208082966.svg)](https://zenodo.org/badge/latestdoi/208082966) [![PyPI version](https://badge.fury.io/py/nngeometry.svg)](https://badge.fury.io/py/nngeometry)



NNGeometry allows you to:
 - compute **Fisher Information Matrices** (FIM) or derivates, using efficient approximations such as low-rank matrices, KFAC, diagonal and so on.
 - compute finite-width **Neural Tangent Kernels** (Gram matrices), even for multiple output functions.
 - compute **per-examples jacobians** of the loss w.r.t network parameters, or of any function such as the network's output.
 - easily and efficiently compute linear algebra operations involving these matrices **regardless of their approximation**.
 - compute **implicit** operations on these matrices, that do not require explicitely storing large matrices that would not fit in memory.

## Example

In the Elastic Weight Consolidation continual learning technique, you want to compute <img src="https://render.githubusercontent.com/render/math?math=\left(\mathbf{w}-\mathbf{w}_{A}\right)^{\top}F\left(\mathbf{w}-\mathbf{w}_{A}\right)">. It can be achieved with a diagonal approximation for the FIM using: 
```python
F = FIM(model=model,
        loader=loader,
        representation=PMatDiag,
        n_output=10)

regularizer = F.vTMv(w - w_a)
```
If diagonal is not sufficiently accurate then you could instead choose a KFAC approximation, by just changing `PMatDiag` to `PMatKFAC` in the above. Note that it internally involves very different operations, depending on the chosen representation (e.g. KFAC, EKFAC, ...).

## Documentation

You can visit the documentation at https://nngeometry.readthedocs.io.

More example usage are available in the repository https://github.com/tfjgeorge/nngeometry-examples.

## Feature requests, bugs or contributions

We welcome any feature request or bug report in the [issue tracker](https://github.com/tfjgeorge/nngeometry/issues).

We also welcome contributions, please submit your PRs!

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

## License

This project is distributed under the MIT license (see LICENSE file).
This project also includes code licensed under the BSD 3 clause as it borrows some code from https://github.com/owkin/grad-cnns.
