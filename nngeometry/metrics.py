from enum import StrEnum
from functools import partial

import torch
from torch.distributions.multivariate_normal import MultivariateNormal

from .backend import TorchFuncJacobianBackend, TorchHooksJacobianBackend
from .layercollection import LayerCollection
from .object.pspace import PMatImplicit


def FIM_MonteCarlo(
    model,
    loader,
    representation,
    variant="classif_logits",
    trials=1,
    device="cpu",
    function=None,
    layer_collection=None,
    verbose=False,
    **kwargs,
):
    """
    Helper that creates a matrix computing the Fisher Information
    Matrix using a Monte-Carlo estimate of y|x with `trials` samples per
    example

    Parameters
    ----------
    model : torch.nn.Module
        The model that contains all parameters of the function
    loader : torch.utils.data.DataLoader
        DataLoader for computing expectation over the input space
    representation : class
        The parameter matrix representation that will be used to store
        the matrix
    variants : string 'classif_logits' or 'regression', optional
            (default='classif_logits')
        Variant to use depending on how you interpret your function.
        Possible choices are:
         - 'classif_logits' when using logits for classification
         - 'classif_logsoftmax' when using log_softmax values for classification
         - 'segmentation_logits' when using logits in a segmentation task
    trials : int, optional (default=1)
        Number of trials for Monte Carlo sampling
    device : string, optional (default='cpu')
        Target device for the returned matrix
    function : function, optional (default=None)
        An optional function if different from `model(input)`. If
        it is different from None, it will override the device
        parameter.
    layer_collection : layercollection.LayerCollection, optional
            (default=None)
        An optional layer collection

    """

    if function is None:

        def function(*d):
            return model(d[0].to(device))

    if layer_collection is None:
        layer_collection = LayerCollection.from_model(model)

    if variant == "classif_logits":

        def fim_function(*d):
            log_softmax = torch.log_softmax(function(*d), dim=1)
            probabilities = torch.exp(log_softmax)
            sampled_targets = torch.multinomial(probabilities, trials, replacement=True)
            return trials**-0.5 * torch.gather(log_softmax, 1, sampled_targets)

    elif variant == "classif_logsoftmax":

        def fim_function(*d):
            log_softmax = function(*d)
            probabilities = torch.exp(log_softmax)
            sampled_targets = torch.multinomial(probabilities, trials, replacement=True)
            return trials**-0.5 * torch.gather(log_softmax, 1, sampled_targets)

    elif variant == "segmentation_logits":

        def fim_function(*d):
            log_softmax = torch.log_softmax(function(*d), dim=1)
            s_mb, s_c, s_h, s_w = log_softmax.size()
            log_softmax = (
                log_softmax.permute(0, 2, 3, 1).contiguous().view(s_mb * s_h * s_w, s_c)
            )
            probabilities = torch.exp(log_softmax)
            sampled_indices = torch.multinomial(probabilities, trials, replacement=True)
            sampled_targets = torch.gather(log_softmax, 1, sampled_indices)
            sampled_targets = sampled_targets.view(s_mb, s_h * s_w, trials).sum(dim=1)
            return trials**-0.5 * sampled_targets

    elif variant == "regression":
        if "covariance" in kwargs:
            sigma_2 = kwargs["covariance"]
        else:
            sigma_2 = 1

        def fim_function(*d):
            output = function(*d)
            covariance = sigma_2 * torch.eye(
                output.size(-1), dtype=output.dtype, device=output.device
            )
            mean = torch.zeros(
                output.size(-1), dtype=output.dtype, device=output.device
            )
            normal = MultivariateNormal(loc=mean, covariance_matrix=covariance).sample(
                sample_shape=(trials, output.size(0))
            )
            return trials**-0.5 * (normal * output).sum(dim=-1)

    else:
        raise NotImplementedError

    generator = TorchHooksJacobianBackend(
        model=model, function=fim_function, verbose=verbose
    )
    return representation(
        generator=generator,
        examples=loader,
        layer_collection=layer_collection,
        **kwargs,
    )


def FIM(
    model,
    loader,
    representation,
    variant="classif_logits",
    device="cpu",
    function=None,
    layer_collection=None,
    verbose=False,
    **kwargs,
):
    """
    Helper that creates a matrix computing the Fisher Information
    Matrix using closed form expressions for the expectation y|x
    as described in (Pascanu and Bengio, 2013)

    Parameters
    ----------
    model : torch.nn.Module
        The model that contains all parameters of the function
    loader : torch.utils.data.DataLoader
        DataLoader for computing expectation over the input space
    representation : class
        The parameter matrix representation that will be used to store
        the matrix
    variant : string 'classif_logits' or 'classif_binary_logits' or
        'regression', optional (default='classif_logits')
        Variant to use depending on how you interpret your neural network.
        Possible choices are:
         - 'classif_logits' the NN returns logits to be used in a softmax
         model.
         - 'classif_binary_logits' the NN returns 1d logits to be used in
         a sigmoid model.
         - 'regression' the NN returns the mean of a normal distribution with
         identity covariance.
    device : string, optional (default='cpu')
        Target device for the returned matrix
    function : function, optional (default=None)
        An optional function if different from `model(input)`. If
        it is different from None, it will override the device
        parameter.
    layer_collection : layercollection.LayerCollection, optional
            (default=None)
        An optional layer collection
    """

    if layer_collection is None:
        layer_collection = LayerCollection.from_model(model)

    if representation == PMatImplicit:

        def function_fim(*d):
            return SQRT_VAR[variant](lambda predictions, _: predictions, *d)

        generator = TorchFuncJacobianBackend(
            model=model, function=function_fim, verbose=verbose
        )

    else:
        if function is None:

            def function(*d):
                return model(d[0].to(device))

        function_fim = partial(SQRT_VAR[variant], function)
        generator = TorchHooksJacobianBackend(
            model=model, function=function_fim, verbose=verbose
        )

    return representation(
        generator=generator,
        examples=loader,
        layer_collection=layer_collection,
        **kwargs,
    )


class FIM_Types(StrEnum):
    CLASSIF_LOGITS = "classif_logits"
    CLASSIF_BINARY_LOGITS = "classif_binary_logits"
    REGRESSION = "regression"


def _proj_to_L_multinomial(x, p, q):
    # n -> n-1
    eps = torch.finfo(p.dtype).eps
    pixi = p * x
    pixi = pixi.sum(dim=1, keepdim=True) - pixi.cumsum(dim=1)
    return x[:, :-1] - pixi[:, :-1] / torch.clip(q[:, :-1], eps, 1)


def _diag_var_multinomial(p, q):
    eps = torch.finfo(p.dtype).eps
    return p[:, :-1] * q[:, :-1] / torch.clip(p[:, :-1] + q[:, :-1], eps, 1)


def _sqrt_var_classif_logits(function, *d):
    # This uses the symbolic expression from:
    # An Exact Cholesky Decomposition and the Generalized Inverse of
    # the Variance-Covariance  Matrix of the Multinomial Distribution,
    # with Applications
    # Kunio Tanabe and Masahiko Sagae 1992

    x = function(*d)  # logits
    p = torch.softmax(x, dim=1).detach()
    q = 1 - p.cumsum(dim=1)

    # Multiply by L
    x_L = _proj_to_L_multinomial(x, p, q)

    # get diagonal
    d = _diag_var_multinomial(p, q)

    x_L = x_L * torch.clip(d, 0, 1) ** 0.5
    return x_L


def _sqrt_var_classif_binary_logits(function, *d):
    logit = function(*d)
    probs = torch.nn.functional.sigmoid(logit).detach()
    coef = torch.sqrt(probs * (1 - probs))
    return logit * coef


def _sqrt_var_regression(function, *d):
    estimates = function(*d)
    return estimates


SQRT_VAR = {
    FIM_Types.CLASSIF_LOGITS: _sqrt_var_classif_logits,
    FIM_Types.CLASSIF_BINARY_LOGITS: _sqrt_var_classif_binary_logits,
    FIM_Types.REGRESSION: _sqrt_var_regression,
}
