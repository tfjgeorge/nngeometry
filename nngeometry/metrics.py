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
                   layer_collection=None):
    """
    Helper that creates a matrix computing the Fisher Information
    Matrix using a Monte-Carlo estimate of y|x with 1 sample per example
    """
    
    if layer_collection is None:
        layer_collection = LayerCollection.from_model(model)

    if variant == 'classif_logits':

        def function(input, target):
            log_softmax = torch.log_softmax(model(input.to(device)), dim=1)
            probabilities = torch.exp(log_softmax)
            sampled_targets = torch.multinomial(probabilities, trials, replacement=True)
            return trials ** -.5 * torch.gather(log_softmax, 1, sampled_targets)
    elif variant == 'classif_logsoftmax':

        def function(input, target):
            log_softmax = model(input.to(device))
            probabilities = torch.exp(log_softmax)
            sampled_targets = torch.multinomial(probabilities, trials, replacement=True)
            return trials ** -.5 * torch.gather(log_softmax, 1, sampled_targets)
    else:
        raise NotImplementedError

    generator = Jacobian(layer_collection=layer_collection,
                         model=model,
                         loader=loader,
                         function=function,
                         n_output=trials)
    return representation(generator)


def FIM(model,
        loader,
        representation,
        n_output,
        variant='classif_logits',
        device='cpu',
        layer_collection=None):
    """
    Helper that creates a matrix computing the Fisher Information
    Matrix using closed form expressions for the expectation y|x
    as described in (Pascanu and Bengio, 2013)
    """
    # TODO: test

    if layer_collection is None:
        layer_collection = LayerCollection.from_model(model)

    if variant == 'classif_logits':

        def function(*d):
            inputs = d[0].to(device)
            log_probs = torch.log_softmax(model(inputs), dim=1)
            probs = torch.exp(log_probs).detach()
            return (log_probs * probs**.5)

    elif variant == 'regression':

        def function(*d):
            inputs = d[0].to(device)
            estimates = model(inputs)
            return estimates
    else:
        raise NotImplementedError

    generator = Jacobian(layer_collection=layer_collection,
                         model=model,
                         loader=loader,
                         function=function,
                         n_output=n_output)
    return representation(generator)
