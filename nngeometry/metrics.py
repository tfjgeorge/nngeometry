from functools import partial
from enum import StrEnum

import torch

from .backend import TorchHooksJacobianBackend
from .layercollection import LayerCollection


def FIM_MonteCarlo(
    model,
    loader,
    representation,
    variant="classif_logits",
    trials=1,
    device="cpu",
    function=None,
    layer_collection=None,
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

    else:
        raise NotImplementedError

    generator = TorchHooksJacobianBackend(
        layer_collection=layer_collection,
        model=model,
        function=fim_function,
    )
    return representation(generator=generator, examples=loader)


def FIM(
    model,
    loader,
    representation,
    variant="classif_logits",
    device="cpu",
    function=None,
    layer_collection=None,
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

    if function is None:

        def function(*d):
            return model(d[0].to(device))

    if layer_collection is None:
        layer_collection = LayerCollection.from_model(model)

    function_fim = partial(SQRT_VAR[variant], function)

    generator = TorchHooksJacobianBackend(
        layer_collection=layer_collection,
        model=model,
        function=function_fim,
    )

    return representation(generator=generator, examples=loader)


class FIM_Types(StrEnum):
    CLASSIF_LOGITS = "classif_logits"
    CLASSIF_BINARY_LOGITS = "classif_binary_logits"
    REGRESSION = "regression"


def _sqrt_var_classif_logits(function, *d):
    logits = function(*d)
    p = torch.softmax(logits, dim=1).detach()
    q = 1 - p.cumsum(dim=1)

    # Multiply by L
    logits_p = logits * p
    pixi = logits_p.sum(dim=1, keepdim=True) - logits_p.cumsum(dim=1)

    logits_L = logits[:, :-1] - pixi[:, :-1] / (q + torch.finfo(q.dtype).eps)[:, :-1]

    d = p[:, :-1] * q[:, :-1] / (p[:, :-1] + q[:, :-1] + torch.finfo(q.dtype).eps)

    x_p = logits_L * d**0.5

    return x_p


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
