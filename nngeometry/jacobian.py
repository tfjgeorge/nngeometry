from nngeometry.object.map import PushForwardDense
from nngeometry.backend import TorchHooksJacobianBackend


def Jacobian(
    model,
    loader,
    function=None,
    representation=PushForwardDense,
    layer_collection=None,
):

    backend = TorchHooksJacobianBackend(
        model=model, function=function, layer_collection=layer_collection
    )

    return representation(generator=backend, examples=loader)
