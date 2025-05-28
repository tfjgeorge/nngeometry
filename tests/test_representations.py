import pytest
import torch
from nngeometry.object.map import PFMapDense
from tasks import get_conv_gn_task, get_conv_task, get_fullyconnect_task
from utils import check_tensors

from nngeometry.backend import TorchHooksJacobianBackend
from nngeometry.object.pspace import PMatBlockDiag, PMatDense, PMatDiag

nonlinear_tasks = [get_conv_gn_task, get_fullyconnect_task, get_conv_task]

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


@pytest.fixture(autouse=True)
def make_test_deterministic():
    torch.manual_seed(1234)
    yield


def test_diag():
    for get_task in nonlinear_tasks:
        loader, lc, parameters, model1, function1 = get_task()
        _, _, _, model2, function2 = get_task()

        generator1 = TorchHooksJacobianBackend(model=model1, function=function1)
        generator2 = TorchHooksJacobianBackend(model=model2, function=function1)
        M_diag1 = PMatDiag(generator=generator1, examples=loader, layer_collection=lc)
        M_diag2 = PMatDiag(generator=generator2, examples=loader, layer_collection=lc)

        prod = M_diag1.mm(M_diag2)

        M_diag1_tensor = M_diag1.to_torch()
        M_diag2_tensor = M_diag2.to_torch()

        prod_tensor = prod.to_torch()

        check_tensors(torch.mm(M_diag1_tensor, M_diag2_tensor), prod_tensor)


def test_dense():
    for get_task in nonlinear_tasks:
        loader, lc, parameters, model1, function1 = get_task()
        _, _, _, model2, function2 = get_task()

        generator1 = TorchHooksJacobianBackend(model=model1, function=function1)
        generator2 = TorchHooksJacobianBackend(model=model2, function=function1)
        M_dense1 = PMatDense(generator=generator1, examples=loader, layer_collection=lc)
        M_dense2 = PMatDense(generator=generator2, examples=loader, layer_collection=lc)

        prod = M_dense1.mm(M_dense2)

        M_dense1_tensor = M_dense1.to_torch()
        M_dense2_tensor = M_dense2.to_torch()

        prod_tensor = prod.to_torch()

        check_tensors(torch.mm(M_dense1_tensor, M_dense2_tensor), prod_tensor)


def test_blockdiag():
    for get_task in nonlinear_tasks:
        loader, lc, parameters, model1, function1 = get_task()
        _, _, _, model2, function2 = get_task()

        generator1 = TorchHooksJacobianBackend(model=model1, function=function1)
        generator2 = TorchHooksJacobianBackend(model=model2, function=function1)
        M_blockdiag1 = PMatBlockDiag(
            generator=generator1, examples=loader, layer_collection=lc
        )
        M_blockdiag2 = PMatBlockDiag(
            generator=generator2, examples=loader, layer_collection=lc
        )

        prod = M_blockdiag1.mm(M_blockdiag2)

        M_blockdiag1_tensor = M_blockdiag1.to_torch()
        M_blockdiag2_tensor = M_blockdiag2.to_torch()

        prod_tensor = prod.to_torch()

        check_tensors(torch.mm(M_blockdiag1_tensor, M_blockdiag2_tensor), prod_tensor)


def test_pfmap():
    for get_task in nonlinear_tasks:
        loader, lc, parameters, model1, function1 = get_task()
        _, _, _, model2, function2 = get_task()

        generator1 = TorchHooksJacobianBackend(model=model1, function=function1)
        generator2 = TorchHooksJacobianBackend(model=model2, function=function1)
        pfmap1 = PFMapDense(generator=generator1, examples=loader, layer_collection=lc)
        pfmap2 = PFMapDense(generator=generator2, examples=loader, layer_collection=lc)

        torch.testing.assert_close(
            pfmap1.to_torch() + pfmap2.to_torch(), (pfmap1 + pfmap2).to_torch()
        )
        torch.testing.assert_close(1.23 * pfmap1.to_torch(), (1.23 * pfmap1).to_torch())
        torch.testing.assert_close(
            pfmap1.to_torch() - pfmap2.to_torch(), (pfmap1 - pfmap2).to_torch()
        )
