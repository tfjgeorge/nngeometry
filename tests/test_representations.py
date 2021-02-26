import torch
from tasks import (get_conv_gn_task, get_fullyconnect_task, get_conv_task)
from nngeometry.object.pspace import (PMatDense, PMatDiag, PMatBlockDiag,
                                      PMatImplicit, PMatLowRank, PMatQuasiDiag)
from nngeometry.generator import Jacobian
from utils import check_ratio, check_tensors, check_angle
import pytest


nonlinear_tasks = [get_conv_gn_task, get_fullyconnect_task, get_conv_task]

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

@pytest.fixture(autouse=True)
def make_test_deterministic():
    torch.manual_seed(1234)
    yield

def test_diag():
    for get_task in nonlinear_tasks:
        loader, lc, parameters, model1, function1, n_output = get_task()
        _, _, _, model2, function2, _ = get_task()

        generator1 = Jacobian(layer_collection=lc,
                              model=model1,
                              function=function1,
                              n_output=n_output)
        generator2 = Jacobian(layer_collection=lc,
                              model=model2,
                              function=function1,
                              n_output=n_output)
        M_diag1 = PMatDiag(generator=generator1, examples=loader)
        M_diag2 = PMatDiag(generator=generator2, examples=loader)

        prod = M_diag1.mm(M_diag2)

        M_diag1_tensor = M_diag1.get_dense_tensor()
        M_diag2_tensor = M_diag2.get_dense_tensor()

        prod_tensor = prod.get_dense_tensor()

        check_tensors(torch.mm(M_diag1_tensor, M_diag2_tensor),
                      prod_tensor)

def test_dense():
    for get_task in nonlinear_tasks:
        loader, lc, parameters, model1, function1, n_output = get_task()
        _, _, _, model2, function2, _ = get_task()

        generator1 = Jacobian(layer_collection=lc,
                              model=model1,
                              function=function1,
                              n_output=n_output)
        generator2 = Jacobian(layer_collection=lc,
                              model=model2,
                              function=function1,
                              n_output=n_output)
        M_dense1 = PMatDense(generator=generator1, examples=loader)
        M_dense2 = PMatDense(generator=generator2, examples=loader)

        prod = M_dense1.mm(M_dense2)

        M_dense1_tensor = M_dense1.get_dense_tensor()
        M_dense2_tensor = M_dense2.get_dense_tensor()

        prod_tensor = prod.get_dense_tensor()

        check_tensors(torch.mm(M_dense1_tensor, M_dense2_tensor),
                      prod_tensor)

def test_blockdiag():
    for get_task in nonlinear_tasks:
        loader, lc, parameters, model1, function1, n_output = get_task()
        _, _, _, model2, function2, _ = get_task()

        generator1 = Jacobian(layer_collection=lc,
                              model=model1,
                              function=function1,
                              n_output=n_output)
        generator2 = Jacobian(layer_collection=lc,
                              model=model2,
                              function=function1,
                              n_output=n_output)
        M_blockdiag1 = PMatBlockDiag(generator=generator1, examples=loader)
        M_blockdiag2 = PMatBlockDiag(generator=generator2, examples=loader)

        prod = M_blockdiag1.mm(M_blockdiag2)

        M_blockdiag1_tensor = M_blockdiag1.get_dense_tensor()
        M_blockdiag2_tensor = M_blockdiag2.get_dense_tensor()

        prod_tensor = prod.get_dense_tensor()

        check_tensors(torch.mm(M_blockdiag1_tensor, M_blockdiag2_tensor),
                      prod_tensor)