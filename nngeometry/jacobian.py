from nngeometry.object.map import PFMapDense
from nngeometry.backend import TorchHooksJacobianBackend


def Jacobian(
    model,
    loader,
    function=None,
    representation=PFMapDense,
    layer_collection=None,
):

    backend = TorchHooksJacobianBackend(
        model=model, function=function, layer_collection=layer_collection
    )

    return representation(generator=backend, examples=loader)
