from nngeometry.generator.jacobian import Jacobian
from nngeometry.object.pspace import PMatBlockDiag, PMatKFAC, PMatEKFAC
import torch
from tasks import get_fullyconnect_task, get_conv_task, device
from nngeometry.object.vector import random_pvector
from utils import check_ratio, check_tensors
import pytest


@pytest.fixture(autouse=True)
def make_test_deterministic():
    torch.manual_seed(1234)
    yield


def test_pspace_ekfac_vs_kfac():
    """
    Check that EKFAC matrix is closer to block diag one in the
    sense of the Frobenius norm
    """
    eps = 1e-4
    for get_task in [get_fullyconnect_task, get_conv_task]:
        loader, lc, parameters, model, function, n_output = get_task()
        model.train()
        generator = Jacobian(layer_collection=lc,
                             model=model,
                             function=function,
                             n_output=n_output)

        M_kfac = PMatKFAC(generator=generator, examples=loader)
        M_ekfac = PMatEKFAC(generator=generator, examples=loader)
        M_blockdiag = PMatBlockDiag(generator=generator, examples=loader)

        # here KFAC and EKFAC should be the same
        for split in [True, False]:
            diff = M_kfac.get_dense_tensor(split_weight_bias=split) - \
                M_ekfac.get_dense_tensor(split_weight_bias=split)
            assert torch.norm(diff) < eps

        # now we compute the exact diagonal:
        M_ekfac.update_diag(loader)
        assert torch.norm(M_kfac.get_dense_tensor()
                          - M_blockdiag.get_dense_tensor()) > \
            torch.norm(M_ekfac.get_dense_tensor()
                       - M_blockdiag.get_dense_tensor())


def test_pspace_ekfac_vs_direct():
    """
    Check EKFAC basis operations against direct computation using
    get_dense_tensor
    """
    for get_task in [get_fullyconnect_task, get_conv_task]:
        loader, lc, parameters, model, function, n_output = get_task()
        model.train()

        generator = Jacobian(layer_collection=lc,
                             model=model,
                             function=function,
                             n_output=n_output)

        M_ekfac = PMatEKFAC(generator=generator, examples=loader)
        v = random_pvector(lc, device=device)

        # the second time we will have called update_diag
        for i in range(2):
            vTMv_direct = torch.dot(torch.mv(M_ekfac.get_dense_tensor(),
                                             v.get_flat_representation()),
                                    v.get_flat_representation())
            vTMv_ekfac = M_ekfac.vTMv(v)
            check_ratio(vTMv_direct, vTMv_ekfac)

            trace_ekfac = M_ekfac.trace()
            trace_direct = torch.trace(M_ekfac.get_dense_tensor())
            check_ratio(trace_direct, trace_ekfac)

            frob_ekfac = M_ekfac.frobenius_norm()
            frob_direct = torch.norm(M_ekfac.get_dense_tensor())
            check_ratio(frob_direct, frob_ekfac)

            mv_direct = torch.mv(M_ekfac.get_dense_tensor(),
                                 v.get_flat_representation())
            mv_ekfac = M_ekfac.mv(v)
            check_tensors(mv_direct, mv_ekfac.get_flat_representation())

            # Test inverse
            regul = 1e-5    
            M_inv = M_ekfac.inverse(regul=regul)
            v_back = M_inv.mv(mv_ekfac + regul * v)
            check_tensors(v.get_flat_representation(),
                          v_back.get_flat_representation())

            # Test solve
            v_back = M_ekfac.solve(mv_ekfac + regul * v, regul=regul)
            check_tensors(v.get_flat_representation(),
                          v_back.get_flat_representation())

            # Test rmul
            M_mul = 1.23 * M_ekfac
            check_tensors(1.23 * M_ekfac.get_dense_tensor(),
                          M_mul.get_dense_tensor())

            M_ekfac.update_diag(loader)