Parameter space representations
===============================

Parameter space representations are :math:`d \times d` objects that define metrics in parameter space such as:

 - Fisher Information Matrices/Gauss-Newton matrix
 - Gradient 2nd moment (e.g. the sometimes called *Empirical Fisher*)
 - Other covariances such as in Bayesian Deep Learning

These matrices are often too large to fit in memory, for instance when :math:`d` is in the order of :math:`10^6 - 10^8`
as is typical in current deep networks. Here is a list of parameter space representations that are available in NNGeometry,
computed on a small network, represented as images where each pixel represent a component of the matrix, and the color is
the magnitude of these components. These matrices are normalized by their diagonal (i.e. these are correlation matrices) for
better visualization:

:class:`nngeometry.object.pspace.PMatDense` representation: this is the usual dense matrix. Memory cost: :math:`d \times d`

.. image:: https://github.com/tfjgeorge/nngeometry/raw/master/docs/repr_img/PMatDense.png
  :width: 400
  
:class:`nngeometry.object.pspace.PMatBlockDiag` representation: a block-diagonal representation where diagonal blocks are
dense matrices corresponding to parameters of a single layer, and cross-layer interactions are ignored (their coefficients are
set to :math:`0`). Memory cost: :math:`\sum_l d_l \times d_l` where :math:`d_l` is the number of parameters of layer :math:`l`.

.. image:: https://github.com/tfjgeorge/nngeometry/raw/master/docs/repr_img/PMatBlockDiag.png
  :width: 400

:class:`nngeometry.object.pspace.PMatKFAC` representation :cite:p:`martens2015optimizing, grosse2016kronecker`: a block-diagonal representation where diagonal blocks are
factored as the Kronecker product of two smaller matrices, and cross-layer interactions are ignored (their coefficients are
set to :math:`0`). Memory cost: :math:`\sum_l g_l \times g_l + a_l \times a_l` where :math:`a_l` is the number of neurons of the
input of layer :math:`l` and :math:`g_l` is the number of pre-activations of the output of layer :math:`l`.

.. image:: https://github.com/tfjgeorge/nngeometry/raw/master/docs/repr_img/PMatKFAC.png
  :width: 400

:class:`nngeometry.object.pspace.PMatEKFAC` representation :cite:p:`george2018fast`: a block-diagonal representation where diagonal blocks are
factored as a diagonal matrix in a Kronecker factored eigenbasis, and cross-layer interactions are ignored (their coefficients are
set to :math:`0`). Memory cost: :math:`\sum_l g_l \times g_l + a_l \times a_l + d_l` where :math:`a_l` is the number of neurons of the
input of layer :math:`l` and :math:`g_l` is the number of pre-activations of the output of layer :math:`l`, and :math:`d_l` is 

.. image:: https://github.com/tfjgeorge/nngeometry/raw/master/docs/repr_img/PMatEKFAC.png
  :width: 400

:class:`nngeometry.object.pspace.PMatDiag` representation: a diagonal representation that ignores all interactions between parameters. 
Memory cost: :math:`d`

.. image:: https://github.com/tfjgeorge/nngeometry/raw/master/docs/repr_img/PMatDiag.png
  :width: 400

:class:`nngeometry.object.pspace.PMatQuasiDiag` representation :cite:p:`ollivier2015riemannian`: a diagonal representation where for each neuron, a coefficient is also
stored that measures the interaction between this neuron's weights and the corresponding bias. 
Memory cost: :math:`2 \times d`

.. image:: https://github.com/tfjgeorge/nngeometry/raw/master/docs/repr_img/PMatQuasiDiag.png
  :width: 400