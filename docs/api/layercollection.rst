Layer collection
================

Layer collections describe the structure of parameters that will be differentiated. We need the LayerCollection object in order to be able to map components of different objects together. As an example, when performing a matrix-vector product using a block diagonal representation, we need to make sure that elements of the vector corresponding to parameters from layer 1 are multiplied with the diagonal block also corresponding to parameters from layer 1, and so on.

Typical use cases include:

 - All parameters of your network: for this you can simply use the constructor :func:`nngeometry.layercollection.LayerCollection.from_model`
 - Only parameters from some layers of your network. In this case you need to
  1. instantiate a new LayerCollection object
  2. add your layers one at a time using :func:`nngeometry.layercollection.LayerCollection.add_layer_from_model`


.. automodule:: nngeometry.layercollection
    :members:
    :undoc-members:
