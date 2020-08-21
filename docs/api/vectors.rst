Vector representations
======================

In NNGeometry, vectors are not just a bunch of scalars, but they have a semantic meaning.

 - :class:`nngeometry.object.vector.PVector` objects are vectors living in the parameter space of a neural network model. An example of such vector is :math:`\delta \mathbf w` in the EWC penalty :math:`\delta \mathbf w^\top F \delta \mathbf w`.
 - :class:`nngeometry.object.vector.FVector` objects are vectors living in the function space of a neural network model. An example of such vector is :math:`\mathbf{f}=\left(f\left(x_{1}\right),\ldots,f\left(x_{n}\right)\right)^{\top}` where :math:`f` is a neural network and :math:`x_1,\ldots,x_n` are examples from a training dataset.


.. automodule:: nngeometry.object.vector
    :members:
    :undoc-members:
    :show-inheritance:
