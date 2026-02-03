import pytest

from nngeometry.object.pspace import PMatEye
from tests.tasks import get_conv_bn_task


def test_inverse_deprecation():
    get_task = get_conv_bn_task

    loader, lc, parameters, model, function = get_task()

    pmat_eye = PMatEye(layer_collection=lc)

    with pytest.warns(DeprecationWarning):
        pmat_eye.inverse()
