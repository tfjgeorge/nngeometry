# NNGeometry

![Build Status](https://github.com/tfjgeorge/nngeometry/actions/workflows/test.yml/badge.svg) [![codecov](https://codecov.io/gh/tfjgeorge/nngeometry/branch/master/graph/badge.svg)](https://codecov.io/gh/tfjgeorge/nngeometry) [![DOI](https://zenodo.org/badge/208082966.svg)](https://zenodo.org/badge/latestdoi/208082966) [![PyPI version](https://badge.fury.io/py/nngeometry.svg)](https://badge.fury.io/py/nngeometry)

NNGeometry allows you to:
 - compute Gauss-Newton or **Fisher Information Matrices** (`FIM` and `FIM_MonteCarlo`), as well as any matrix that is written as the covariance of gradients w.r.t. parameters, using efficient approximations such as low-rank matrices, KFAC, EKFAC, diagonal and so on. Some of these representations also work for hessians (`Hessian`).
 - compute finite-width **Neural Tangent Kernels** evaluated on a set of examples (`GramMatrix`), even for multiple output functions.
 - compute **per-examples jacobians** of the loss w.r.t network parameters (`Jacobian`), or of any function such as the network's output.
 - easily and efficiently compute linear algebra operations involving these matrices **regardless of their approximation**.
 - compute **implicit** operations on these matrices, that do not require explicitely storing large matrices that would not fit in memory.

It offers a high level abstraction over the parameter and function spaces described by neural networks. As a simple example, a parameter space vector `PVector` actually contains weight matrices, bias vectors, or convolutions kernels of the whole neural network (a set of tensors). Using NNGeometry's API, performing a step in parameter space (e.g. an update of your favorite optimization algorithm) is abstracted as a python addition: `w_next = w_previous + epsilon * delta_w`.

## Example

In the Elastic Weight Consolidation continual learning technique, you want to compute $`\left(\mathbf{w}-\mathbf{w}_{A}\right)^{\top}F\left(\mathbf{w}-\mathbf{w}_{A}\right)`$. It can be achieved with a diagonal approximation for the FIM using: 
```python
F = FIM(model=model,
        loader=loader,
        representation=PMatDiag)

regularizer = F.vTMv(w - w_a)
```
The first statement instantiates a diagonal matrix, and populates it with the diagonal coefficients of the FIM of the model `model` computed using the examples from the dataloader `loader`.

If diagonal is not sufficiently accurate then you could instead choose a KFAC approximation, by just changing `PMatDiag` to `PMatKFAC` in the above. Note that it internally involves very different operations, depending on the chosen representation (e.g. KFAC, EKFAC, ...).

## Documentation

You can visit the documentation at https://nngeometry.readthedocs.io.

More example usage are available in the repository https://github.com/tfjgeorge/nngeometry-examples.

## Feature requests, bugs, contributions, or any kind of request

You are now many who are using NNGeometry in your work: do not hesitate to drop me a line (tfjgeorge@gmail.com) about your project so that I have a better understanding of your use cases or the current limitations of the library.

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
  version      = {v0.3},
  doi          = {10.5281/zenodo.4532597},
  url          = {https://doi.org/10.5281/zenodo.4532597}
}
```

## License

This project is distributed under the MIT license (see LICENSE file).
This project also includes code licensed under the BSD 3 clause as it borrows some code from https://github.com/owkin/grad-cnns.
