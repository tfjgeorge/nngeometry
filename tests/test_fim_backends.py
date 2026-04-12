import math

import pytest
import torch
from tasks import (
    device,
    get_conv_task,
    get_fullyconnect_onlylast_task,
    get_fullyconnect_task,
    get_linear_conv_task,
    get_linear_fc_task,
    to_device,
)
from utils import update_model

from nngeometry import FIM, Hessian
from nngeometry.backend.torch_func_jacobian import TorchFuncJacobianBackend
from nngeometry.backend.torch_hooks.torch_hooks import TorchHooksJacobianBackend
from nngeometry.object.map import random_pfmap
from nngeometry.object.pspace import PMatDense, PMatImplicit
from nngeometry.object.vector import PVector, random_pvector
from nngeometry.utils import grad

linear_tasks = [
    get_linear_fc_task,
    get_linear_conv_task,
    get_fullyconnect_onlylast_task,
]
nonlinear_tasks = [get_fullyconnect_task, get_conv_task]


@pytest.fixture(autouse=True)
def make_test_deterministic():
    torch.manual_seed(1234)
    yield


def test_torch_hooks_vs_torch_func_fim():
    for get_task in linear_tasks + nonlinear_tasks:
        loader, lc, parameters, model, function = get_task()
        model.train()

        F_hook = PMatImplicit(
            generator=TorchHooksJacobianBackend(
                model=model, function=lambda inputs, _: model(inputs)
            ),
            examples=loader,
            layer_collection=lc,
        )

        F_func = PMatImplicit(
            generator=TorchFuncJacobianBackend(
                model=model, function=lambda predictions, _: predictions
            ),
            examples=loader,
            layer_collection=lc,
        )

        dw = random_pvector(lc)
        torch.testing.assert_close(F_hook.mv(dw).to_torch(), F_func.mv(dw).to_torch())
        assert math.isclose(
            F_hook.vTMv(dw).item(), F_func.vTMv(dw).item(), abs_tol=1e-9
        )

        x = random_pfmap(lc, (10, 3))
        torch.testing.assert_close(F_hook.mmap(x).to_torch(), F_func.mmap(x).to_torch())
        torch.testing.assert_close(F_hook.mapTMmap(x), F_func.mapTMmap(x))
