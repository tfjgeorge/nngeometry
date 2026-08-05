import pytest
import torch
from tasks import get_vit_task

from nngeometry.backend.torch_func_jacobian import TorchFuncJacobianBackend
from nngeometry.backend.torch_hooks.torch_hooks import TorchHooksJacobianBackend
from nngeometry.object.pspace import PMatImplicit
from nngeometry.object.vector import random_pvector


@pytest.fixture(autouse=True)
def make_test_deterministic():
    torch.manual_seed(1234)
    yield


def test_nn_mha_not_supported():
    with pytest.raises(Exception, match="do with layer MultiheadAttention"):
        get_vit_task(torch_attention=True, ignore_unsupported_layers=False)


def test_mha_hooks_vs_func():
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
