# NNGeometry

[![Build Status](https://travis-ci.org/tfjgeorge/nngeometry.svg?branch=master)](https://travis-ci.org/tfjgeorge/nngeometry)

NNGeometry allows you to:
 - compute **Fisher Information Matrices** (FIM) or derivates, using efficient approximatins such as low-rank matrices, KFAC, diagonal and so on
 - compute finite **Neural Tangent Kernels**, even for multiple output functions
 - easily and efficiently compute linear algebra operations involving these matrices **regardless of their approximation**

Example: in the Elastic Weight Consolidation continual learning technique, you want to compute <img src="https://render.githubusercontent.com/render/math?math=\left(\mathbf{w}-\mathbf{w}_{A}\right)^{\top}F\left(\mathbf{w}-\mathbf{w}_{A}\right)">. It can be achieved with a block diagonal approximation for the FIM using: 
```python
F = FIM(model=model,
        loader=loader,
        representation=PMatBlockDiag,
        n_output=10)

regularizer = F.vTMv(w - w_a)
```
If block diagonal is not sufficiently accurate then you could instead choose a KFAC approximation, by just changing `PMatBlockDiag` to `PMatKFAC` in the above.

For more examples, you can visit the documentation at https://nngeometry.readthedocs.io
