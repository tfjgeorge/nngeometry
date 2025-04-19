from nngeometry.backend import TorchHooksJacobianBackend
from nngeometry.layercollection import LayerCollection
from nngeometry.object.map import PFMapDense


def Jacobian(
    model,
    loader,
    function=None,
    representation=PFMapDense,
    layer_collection=None,
):

    backend = TorchHooksJacobianBackend(model=model, function=function)

    if layer_collection is None:
        layer_collection = LayerCollection.from_model(model)

    return representation(
        generator=backend, examples=loader, layer_collection=layer_collection
    )
