import pytest

from nngeometry.backend.torch_hooks.torch_hooks import TorchHooksJacobianBackend
from nngeometry.object.fspace import FMatDense
from nngeometry.object.pspace import PMatEye
from tests.tasks import get_conv_task


def test_frobenius_norm_deprecation():
    get_task = get_conv_task

    loader, lc, parameters, model, function = get_task()

    pmat_eye = PMatEye(layer_collection=lc)
    generator = TorchHooksJacobianBackend(model)
    gram = FMatDense(lc, generator, examples=loader)

    with pytest.warns(DeprecationWarning):
        pmat_eye.frobenius_norm()

    with pytest.warns(DeprecationWarning):
        gram.frobenius_norm()
