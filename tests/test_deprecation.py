import pytest

from nngeometry.backend.torch_hooks.torch_hooks import TorchHooksJacobianBackend
from nngeometry.object.fspace import FMatDense
from nngeometry.object.pspace import PMatEye, PMatKFAC, PMatQuasiDiag
from tests.tasks import get_conv_task


def test_inverse_deprecation():
    loader, lc, parameters, model, function = get_conv_task()

    pmat_eye = PMatEye(layer_collection=lc)
    generator = TorchHooksJacobianBackend(model)

    pmat_kfac = PMatKFAC(lc, generator, examples=loader)
    pmat_qd = PMatQuasiDiag(lc, generator, examples=loader)

    with pytest.warns(DeprecationWarning):
        pmat_eye.inverse()
    with pytest.warns(DeprecationWarning):
        pmat_kfac.inverse()
        pmat_kfac.inverse(use_pi=False)
    with pytest.raises(NotImplementedError):
        pmat_qd.inverse()


def test_frobenius_norm_deprecation():
    loader, lc, parameters, model, function = get_conv_task()

    pmat_eye = PMatEye(layer_collection=lc)
    generator = TorchHooksJacobianBackend(model)

    gram = FMatDense(lc, generator, examples=loader)

    with pytest.warns(DeprecationWarning):
        pmat_eye.frobenius_norm()

    with pytest.warns(DeprecationWarning):
        gram.frobenius_norm()
