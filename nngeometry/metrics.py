from functools import partial
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
    variants : string 'classif_logits' or 'regression', optional
            (default='classif_logits')
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

    if variant == "classif_logits":
        # This uses "An Exact Cholesky Decomposition and the
        # Generalized Inverse of the Variance-Covariance Matrix
        # of the Multinomial Distribution, with Applications"
        # Tanabe and Sagae, 1992

        def function_fim(*d, tri_cache):
            logits = function(*d)
            device = logits.device
            n_out = logits.size(1)
            p = torch.softmax(logits, dim=1).detach()
            q = 1 - p.cumsum(dim=1)
            d = (
                p[:, :-1]
                * q[:, :-1]
                / torch.cat(
                    (torch.ones(size=(q.size(0), 1), device=device), q[:, :-2]), dim=1
                )
            )

            # TODO this allocates memory (once since it is cached)
            # for the only purpose of performing a mm with a triangular matrix
            # -> replace with trmm when it is wrapped in torch
            if len(tri_cache) == 0:
                tri_cache.append(
                    torch.tril(
                        torch.ones(size=(n_out, n_out), device=device), diagonal=-1
                    )
                )
            tri = tri_cache[0]

            x_p = logits * p
            x_p = torch.mm(x_p, tri)
            x_p = x_p[:, :-1]
            x_p = x_p / (q[:, :-1] + torch.finfo().eps)  # avoid divide by 0
            x_p = logits[:, :-1] - x_p
            x_p = x_p * d**0.5

            return x_p

        # caches the tri allocation, this should be automatically GCed
        tri_cache = []

        function_fim = partial(function_fim, tri_cache=tri_cache)

    elif variant == "classif_binary_logits":

        # derivation is in log_sigm.lyx
        def function_fim(*d):
            logit = function(*d)
            probs = torch.nn.functional.sigmoid(logit).detach()
            coef = torch.sqrt(probs * (1 - probs))
            return logit * coef

    elif variant == "regression":

        def function_fim(*d):
            estimates = function(*d)
            return estimates

    else:
        raise NotImplementedError

    generator = TorchHooksJacobianBackend(
        layer_collection=layer_collection,
        model=model,
        function=function_fim,
    )
    
    return representation(generator=generator, examples=loader)
