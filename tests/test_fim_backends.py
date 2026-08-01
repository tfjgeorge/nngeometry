import math
from functools import partial

import pytest
import torch
from tasks import (
    get_conv_task,
    get_fullyconnect_onlylast_task,
    get_fullyconnect_task,
    get_linear_conv_task,
    get_linear_fc_task,
    get_vit_task,
)

from nngeometry.backend.torch_func_jacobian import TorchFuncJacobianBackend
from nngeometry.backend.torch_hooks.torch_hooks import TorchHooksJacobianBackend
from nngeometry.object.map import random_pfmap
from nngeometry.object.pspace import PMatImplicit
from nngeometry.object.vector import random_pvector

linear_tasks = [
    get_linear_fc_task,
    get_linear_conv_task,
    get_fullyconnect_onlylast_task,
]
nonlinear_tasks = [
    get_fullyconnect_task,
    get_conv_task,
    get_vit_task,
    partial(get_vit_task, torch_attention=True),
]


@pytest.fixture(autouse=True)
def make_test_deterministic():
    torch.manual_seed(1234)
    yield


@pytest.mark.parametrize("task", linear_tasks + nonlinear_tasks)
def test_torch_hooks_vs_torch_func_fim(task):
    loader, lc, parameters, model, function = task()
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
    assert math.isclose(F_hook.vTMv(dw).item(), F_func.vTMv(dw).item(), abs_tol=1e-9)

    x = random_pfmap(lc, (10, 3))
    torch.testing.assert_close(F_hook.mmap(x).to_torch(), F_func.mmap(x).to_torch())
    torch.testing.assert_close(F_hook.mapTMmap(x), F_func.mapTMmap(x))
