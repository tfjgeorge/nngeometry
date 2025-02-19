from .generator import HessianTorch


def Hessian(
    model,
    loader,
    representation,
    function=None,
    layer_collection=None,
):

    generator = HessianTorch(
        layer_collection=layer_collection, model=model, function=function
    )
    return representation(generator=generator, examples=loader)
