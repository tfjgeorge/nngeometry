Quick example
=============

With NNGeometry, you can easily manipulate :math:`d \times d` matrices and :math:`d` vectors where :math:`d` is the number of parameter of your neural network, for modern neural networks where :math:`d` can be as big as :math:`10^8`. These matrices include for instance:

 - The *Fisher Information Matrix* (FIM) used in statistics, in the natural gradient algorithm, or as an approximate of the Hessian matrix in some applications.
 - *Posterior covariances* in Bayesian Deep Learning.

You can also compute finite *tangent kernels*.

A naive computation of the FIM would require storing :math:`d \times d` scalars in memory. This is prohibitively large for modern neural network architectures, and a line of research has focused at finding lower memory intensive approximations specific to neural networks, such as KFAC, EKFAC, low-rank approximations, etc. This library proposes a common interface for manipulating these different approximations, called *representations*.

Let us now illustrate this by computing the FIM using the KFAC representation.

   >>> F_kfac = FIM(model=model,
                    loader=loader,
                    representation=PMatKFAC,
                    n_output=10,
                    variant='classif_logits',
                    device='cuda')
   >>> print(F_kfac.trace())

Computing the FIM requires the following arguments:

 - The :class:`torch.nn.Module` ``model`` object is the PyTorch model used as our neural network.
 - The :class:`torch.utils.data.DataLoader` ``loader`` object is the dataloader that contains examples used for computing the FIM.
 - The :class:`.object.PMatKFAC` ``PMatKFAC`` argument specifies which representation to use in order to store the FIM.

 We will next define a vector in parameter space, by using the current value given by our model:

         >>> v = PVector.from_model(model)

 We can now compute the matrix-vector product :math:`F v` by simply calling:

        >>> Fv = F_kfac.mv(v)

 Note that switching from the :class:`.object.PMatKFAC` representation to any other representation such as :class:`.object.PMatDense` is as simple as passing ``representation=PMatDense`` when building the ``F_kfac`` object.
 
 More examples
 =============
 
 More notebook examples can be found at https://github.com/tfjgeorge/nngeometry/tree/master/examples
