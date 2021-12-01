from nngeometry.generator import Jacobian, jacobian
from nngeometry.object.pspace import PMatDense
from tasks import get_conv_task
from utils import check_tensors

def test_conv_impl_switch():
    loader, lc, parameters, model, function, n_output = get_conv_task()
    generator = Jacobian(layer_collection=lc,
                         model=model,
                         function=function,
                         n_output=n_output)

    with jacobian.use_unfold_impl_for_convs():
        PMat_dense_unfold = PMatDense(generator=generator,
                                        examples=loader)

    with jacobian.use_conv_impl_for_convs():
        PMat_dense_conv = PMatDense(generator=generator,
                                    examples=loader)

    check_tensors(PMat_dense_unfold.get_dense_tensor(),
                  PMat_dense_conv.get_dense_tensor())