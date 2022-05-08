import torch
from torch.nn.functional import softmax
from .generator.jacobian import Jacobian
from .layercollection import LayerCollection


def FIM_MonteCarlo(model,
                   loader,
                   representation,
                   variant='classif_logits',
                   trials=1,
                   device='cpu',
                   function=None,
                   layer_collection=None):
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

    if variant == 'classif_logits':

        def fim_function(*d):
            log_softmax = torch.log_softmax(function(*d), dim=1)
            probabilities = torch.exp(log_softmax)
            sampled_targets = torch.multinomial(probabilities, trials,
                                                replacement=True)
            return trials ** -.5 * torch.gather(log_softmax, 1,
                                                sampled_targets)
    elif variant == 'classif_logsoftmax':

        def fim_function(*d):
            log_softmax = function(*d)
            probabilities = torch.exp(log_softmax)
            sampled_targets = torch.multinomial(probabilities, trials,
                                                replacement=True)
            return trials ** -.5 * torch.gather(log_softmax, 1,
                                                sampled_targets)
    elif variant == 'segmentation_logits':

        def fim_function(*d):
            log_softmax = torch.log_softmax(function(*d), dim=1)
            s_mb, s_c, s_h, s_w = log_softmax.size()
            log_softmax = log_softmax.permute(0, 2, 3, 1).contiguous() \
                .view(s_mb * s_h * s_w, s_c)
            probabilities = torch.exp(log_softmax)
            sampled_indices = torch.multinomial(probabilities, trials,
                                                replacement=True)
            sampled_targets = torch.gather(log_softmax, 1,
                                        sampled_indices)
            sampled_targets = sampled_targets.view(s_mb, s_h * s_w, trials) \
                .sum(dim=1)
            return trials ** -.5 * sampled_targets
                                                
    else:
        raise NotImplementedError

    generator = Jacobian(layer_collection=layer_collection,
                         model=model,
                         function=fim_function,
                         n_output=trials)
    return representation(generator=generator, examples=loader)


def FIM(model,
        loader,
        representation,
        n_output,
        variant='classif_logits',
        device='cpu',
        function=None,
        layer_collection=None):
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
    n_output : int
        Number of outputs of the model
    variants : string 'classif_logits' or 'regression', optional
            (default='classif_logits')
        Variant to use depending on how you interpret your function.
        Possible choices are:
         - 'classif_logits' when using logits for classification
         - 'regression' when using a gaussian regression model
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

    if variant == 'classif_logits':

        def function_fim(*d):
            log_probs = torch.log_softmax(function(*d), dim=1)
            probs = torch.exp(log_probs).detach()
            return (log_probs * probs**.5)

    elif variant == 'regression':

        def function_fim(*d):
            estimates = function(*d)
            return estimates
    else:
        raise NotImplementedError

    generator = Jacobian(layer_collection=layer_collection,
                         model=model,
                         function=function_fim,
                         n_output=n_output)
    return representation(generator=generator, examples=loader)
