from nngeometry.backend import TorchHooksJacobianBackend
from nngeometry.layercollection import LayerCollection
from nngeometry.object import FMatDense


def GramMatrix(
    model,
    loader,
    function=None,
    representation=FMatDense,
    layer_collection=None,
):

    backend = TorchHooksJacobianBackend(model=model, function=function)

    if layer_collection is None:
        layer_collection = LayerCollection.from_model(model)

    return representation(
        generator=backend, examples=loader, layer_collection=layer_collection
    )
