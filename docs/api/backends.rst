Backends
========

The spirit of NNGeometry is that you do not directly manipulate backends objects, these can be considered as a backend that you do not need to worry about once instantiated. You instead instantiate concrete representations such as `PMatDense` or `PMatKFAC` and directly call linear algebra operations on these concrete representations.

.. automodule:: nngeometry.backend.torch_hooks
    :members:
