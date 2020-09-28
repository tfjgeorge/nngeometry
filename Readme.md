# NNGeometry

[![Build Status](https://travis-ci.org/tfjgeorge/nngeometry.svg?branch=master)](https://travis-ci.org/tfjgeorge/nngeometry) [![codecov](https://codecov.io/gh/tfjgeorge/nngeometry/branch/master/graph/badge.svg)](https://codecov.io/gh/tfjgeorge/nngeometry)

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
