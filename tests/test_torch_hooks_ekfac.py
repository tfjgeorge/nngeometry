import pytest
import torch
from nngeometry.object.map import PFMapDense
from tasks import device, get_conv_task, get_conv1d_task, get_fullyconnect_task
from utils import check_ratio, check_tensors

from nngeometry.backend import TorchHooksJacobianBackend
from nngeometry.object.pspace import PMatBlockDiag, PMatEKFAC, PMatKFAC
from nngeometry.object.vector import random_pvector


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
    for get_task in [get_conv1d_task, get_fullyconnect_task, get_conv_task]:
        loader, lc, parameters, model, function = get_task()
        model.train()
        generator = TorchHooksJacobianBackend(
            layer_collection=lc, model=model, function=function
        )

        M_kfac = PMatKFAC(generator=generator, examples=loader)
        M_ekfac = PMatEKFAC(generator=generator, examples=loader)
        M_blockdiag = PMatBlockDiag(generator=generator, examples=loader)

        # here KFAC and EKFAC should be the same
        for split in [True, False]:
            diff = M_kfac.to_torch(split_weight_bias=split) - M_ekfac.to_torch(
                split_weight_bias=split
            )
            assert torch.norm(diff) < eps

        # now we compute the exact diagonal:
        M_ekfac.update_diag(loader)
        assert torch.norm(M_kfac.to_torch() - M_blockdiag.to_torch()) > torch.norm(
            M_ekfac.to_torch() - M_blockdiag.to_torch()
        )


def test_pspace_ekfac_vs_direct():
    """
    Check EKFAC basis operations against direct computation using
    to_torch
    """
    for get_task in [get_conv1d_task, get_fullyconnect_task, get_conv_task]:
        loader, lc, parameters, model, function = get_task()
        model.train()

        generator = TorchHooksJacobianBackend(
            layer_collection=lc, model=model, function=function
        )

        M_ekfac = PMatEKFAC(generator=generator, examples=loader)
        v = random_pvector(lc, device=device, dtype=torch.double)

        # the second time we will have called update_diag
        for i in range(2):

            vTMv_direct = torch.dot(
                torch.mv(M_ekfac.to_torch(), v.to_torch()),
                v.to_torch(),
            )
            vTMv_ekfac = M_ekfac.vTMv(v)
            check_ratio(vTMv_direct, vTMv_ekfac)

            trace_ekfac = M_ekfac.trace()
            trace_direct = torch.trace(M_ekfac.to_torch())
            check_ratio(trace_direct, trace_ekfac)

            frob_ekfac = M_ekfac.frobenius_norm()
            frob_direct = torch.norm(M_ekfac.to_torch())
            check_ratio(frob_direct, frob_ekfac)

            mv_direct = torch.mv(M_ekfac.to_torch(), v.to_torch())
            mv_ekfac = M_ekfac.mv(v)
            check_tensors(mv_direct, mv_ekfac.to_torch())

            # Test pow
            check_tensors(
                (M_ekfac**2).to_torch(),
                torch.mm(M_ekfac.to_torch(), M_ekfac.to_torch()),
            )
            check_tensors(
                torch.mm(
                    (M_ekfac ** (1 / 3)).to_torch(),
                    (M_ekfac ** (2 / 3)).to_torch(),
                ),
                M_ekfac.to_torch(),
            )

            # Test inverse
            regul = 1e-5
            M_inv = M_ekfac.inverse(regul=regul)
            v_back = M_inv.mv(mv_ekfac + regul * v)
            check_tensors(v.to_torch(), v_back.to_torch())

            # Test solve with vector
            v_back = M_ekfac.solve(mv_ekfac + regul * v, regul=regul)
            check_tensors(v.to_torch(), v_back.to_torch())

            # Test solve with jacobian
            # TODO improve
            c = 1.678
            stacked_mv = torch.stack([c**i * mv_direct for i in range(6)]).reshape(
                2, 3, -1
            )
            stacked_v = torch.stack([c**i * v.to_torch() for i in range(6)]).reshape(
                2, 3, -1
            )
            jaco = PFMapDense(
                generator=generator,
                data=stacked_mv + regul * stacked_v,
            )
            J_back = M_ekfac.solve(jaco, regul=regul)

            check_tensors(
                stacked_v,
                J_back.to_torch(),
            )

            # Test rmul
            M_mul = 1.23 * M_ekfac
            check_tensors(1.23 * M_ekfac.to_torch(), M_mul.to_torch())

            M_ekfac.update_diag(loader)
