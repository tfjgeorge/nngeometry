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


def test_torch_hooks_vs_torch_func_fim():
    with pytest.raises(Exception, match="do with layer MultiheadAttention"):
        get_vit_task(torch_attention=True, ignore_unsupported_layers=False)
    for torch_attention in [True, False]:
        loader, lc, parameters, model, function = get_vit_task(
            torch_attention=torch_attention
        )
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
