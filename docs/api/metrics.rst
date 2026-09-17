Metrics
=======

The :mod:`nngeometry.metrics` module provides functions to compute various information-theoretic and curvature-related metrics for neural networks, with a focus on the Fisher Information Matrix (FIM) and gradient statistics.

Overview
--------

NNGeometry offers several methods to compute the Fisher Information Matrix and related quantities:

* **FIM** - Exact computation using closed-form expressions (Pascanu and Bengio, 2013)
* **FIM_MonteCarlo** - Monte Carlo estimation using samples from the model's predictive distribution
* **GradientSecondMoment** - Second moment of gradients (also known as the empirical Fisher)

All functions accept a ``representation`` parameter that determines how the resulting matrix is stored (e.g., dense, diagonal, KFAC, etc.). See :doc:`pspace-representations` for available representations.

.. automodule:: nngeometry.metrics
    :members:
    :undoc-members:
    :show-inheritance: