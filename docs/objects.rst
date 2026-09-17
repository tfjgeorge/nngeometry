Core Objects
============

NNGeometry provides high-level functions to compute and manipulate key mathematical objects in neural networks. All are accessible from the top-level package:

.. code-block:: python

    from nngeometry import FIM, FIM_MonteCarlo, GradientSecondMoment, Hessian, Jacobian, GramMatrix

Main objects
------------

Parameter space
~~~~~~~~~~~~~~~ 

**Fisher Information Matrix (FIM)**
  Curvature of the log-likelihood. Two computation modes:
  
  - :func:`nngeometry.metrics.FIM` — Exact, closed-form (Pascanu & Bengio, 2013)
  - :func:`nngeometry.metrics.FIM_MonteCarlo` — Monte Carlo estimate

**Hessian**
  Second derivatives of a loss function. Computed via :func:`nngeometry.hessian.Hessian`.

**Gradient Second Moment**
  Empirical Fisher (second moment of gradients). Computed via :func:`nngeometry.metrics.GradientSecondMoment`.

Function space
~~~~~~~~~~~~~~

**Gram Matrix**
  :math:`J J^\top` in function space. Computed via :func:`nngeometry.gram.GramMatrix`.

Parameter space to function space maps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Jacobian**
  First derivatives of model outputs w.r.t. parameters. Computed via :func:`nngeometry.jacobian.Jacobian`.


Matrix Representations
----------------------

Each object can be instantiated in different **representations** that trade off accuracy for memory/compute:

.. list-table::
   :header-rows: 1

   * - Representation
     - Memory
     - Accuracy
   * - ``PMatDense`` / ``FMatDense``
     - :math:`O(d^2)`
     - Exact
   * - ``PMatDiag``
     - :math:`O(d)`
     - Diagonal only
   * - ``PMatBlockDiag``
     - :math:`O(\sum d_i^2)`
     - Block-diagonal
   * - ``PMatKFAC`` / ``PMatEKFAC``
     - :math:`O(\sum d_i^{in 2} + d_i^{out 2})`
     - Kronecker-factored
   * - ``PMatImplicit``
     - No matrix is formed in memory
     - Exact (matrix-free)
   * - ``PMatLowRank``
     - :math:`O(n d)`
     - Low-rank (rank = n_examples)

Specify the representation via the ``representation`` argument:

.. code-block:: python

    from nngeometry.object import PMatDense, PMatKFAC, PMatImplicit

    # Dense FIM (exact, for small models)
    fim_dense = FIM(model, loader, representation=PMatDense)

    # KFAC FIM (efficient for large models)
    fim_kfac = FIM(model, loader, representation=PMatKFAC)

    # Implicit FIM (matrix-free, for very large models)
    fim_impl = FIM(model, loader, representation=PMatImplicit)

Common Operations
-----------------

All matrix objects support a unified interface:

.. code-block:: python

    # Matrix-vector product
    v = fim.mv(pvector)

    # Quadratic form v^T M v
    quad = fim.vTMv(pvector)

    # Linear solve M x = b
    x = fim.solve(pvector, regul=1e-3)

    # Trace, norm, diagonal
    tr = fim.trace()
    n = fim.norm()
    diag = fim.get_diag()

    # Arithmetic
    M_sum = fim1 + fim2
    M_scaled = 2.0 * fim

    # Inverse (where available)
    fim_inv = fim.inv(regul=1e-3)


See Also
--------

- :doc:`metrics` — FIM and GradientSecondMoment details
- :doc:`pspace-representations` — Parameter space matrix representations
- :doc:`layercollection` — Describing parameter space structure