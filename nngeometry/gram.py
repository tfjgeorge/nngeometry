from nngeometry.object import FMatDense
from nngeometry.backend import TorchHooksJacobianBackend


def GramMatrix(
    model,
    loader,
    function=None,
    representation=FMatDense,
    layer_collection=None,
):

    backend = TorchHooksJacobianBackend(
        model=model, function=function, layer_collection=layer_collection
    )

    return representation(generator=backend, examples=loader)
