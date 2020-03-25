import torch
from .generator.jacobian import Jacobian


def FIM_MonteCarlo1(layer_collection,
                    model,
                    loader,
                    representation,
                    variant='classif_logsoftmax'):
    """
    Helper to create a matrix computing the Fisher Information
    Matrix using a Monte-Carlo estimate with 1 sample per example
    """

    if variant == 'classif_logsoftmax':

        def function(input, target):
            log_softmax = model(input)
            probabilities = torch.exp(log_softmax)
            sampled_targets = torch.multinomial(probabilities, 1)
            return torch.gather(log_softmax, 1, sampled_targets)

        generator = Jacobian(layer_collection=layer_collection,
                             model=model,
                             loader=loader,
                             function=function,
                             n_output=1)
        return representation(generator)
    else:
        raise NotImplementedError
