import pytest

from nngeometry.backend.torch_hooks.torch_hooks import TorchHooksJacobianBackend
from nngeometry.object.pspace import PMatEye, PMatKFAC
from tests.tasks import get_conv_task


def test_inverse_deprecation():
    get_task = get_conv_task

    loader, lc, parameters, model, function = get_task()

    pmat_eye = PMatEye(layer_collection=lc)
    generator = TorchHooksJacobianBackend(model)
    pmat_kfac = PMatKFAC(lc, generator, examples=loader)

    with pytest.warns(DeprecationWarning):
        pmat_eye.inverse()
    with pytest.warns(DeprecationWarning):
        pmat_kfac.inverse()
        pmat_kfac.inverse(use_pi=False)
