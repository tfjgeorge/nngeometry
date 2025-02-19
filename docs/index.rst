.. NNGeometry documentation master file, created by
   sphinx-quickstart on Tue Nov 19 11:02:30 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

NNGeometry: FIMs and NTKs in PyTorch
======================================

NNGeometry is a library built on top of PyTorch aiming at giving tools
to easily manipulate and study properties of Fisher Information Matrices and finite width tangent kernels.

You can start by looking at the quick start example below. Convinced? Then :doc:`install NNGeometry</install>`, try
the tutorials or explore the API reference.

.. warning::
        NNGeometry is under developement, as such it is possible that core components change when
        between versions.

Quick example
=============

Computing the Fisher Information Matrix on a given PyTorch model using a KFAC representation, and then computing its trace is as simple as:

   >>> F_kfac = FIM(model=model,
                    loader=loader,
                    representation=PMatKFAC,
                    variant='classif_logits')
   >>> print(F_kfac.trace())

If we instead wanted to choose a :class:`nngeometry.object.pspace.PMatBlockDiag` representation, we can just replace ``representation=PMatKFAC`` with ``representation=PMatBlockDiag`` in the above.

This example is further detailed in :doc:`/quick_example`. Other available parameter space representations are listed in :doc:`/pspace_repr`.

More examples
=============

More notebook examples can be found at https://github.com/tfjgeorge/nngeometry/tree/master/examples

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

In-depth
========

.. toctree::
   quick_example.rst
   install.rst
   pspace_repr.rst
   :maxdepth: 1

   api/index.rst

References
==========

.. bibliography::
