from nngeometry.layercollection import LayerCollection

from .backend import TorchFuncHessianBackend


def Hessian(
    model,
    loader,
    representation,
    function=None,
    layer_collection=None,
    verbose=False,
):
    generator = TorchFuncHessianBackend(model=model, function=function, verbose=verbose)

    if layer_collection is None:
        layer_collection = LayerCollection.from_model(model)

    return representation(
        generator=generator,
        examples=loader,
        layer_collection=layer_collection,
    )
