import pytest
import torch
from tasks import (
    device,
    get_conv1d_task,
    get_conv_task,
    get_embedding_task,
    get_fullyconnect_task,
    get_linear_3d_task,
)
from utils import check_ratio

from nngeometry.backend import TorchHooksJacobianBackend
from nngeometry.object.map import PFMapDense
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
    for get_task in [
        get_linear_3d_task,
        get_embedding_task,
        get_conv1d_task,
        get_fullyconnect_task,
        get_conv_task,
    ]:
        loader, lc, parameters, model, function = get_task()
        model.train()
        generator = TorchHooksJacobianBackend(model=model, function=function)

        M_kfac = PMatKFAC(generator=generator, examples=loader, layer_collection=lc)
        M_ekfac = PMatEKFAC(generator=generator, examples=loader, layer_collection=lc)
        M_blockdiag = PMatBlockDiag(
            generator=generator, examples=loader, layer_collection=lc
        )

        # here KFAC and EKFAC should be the same
        for split in [True, False]:
            torch.testing.assert_close(
                M_kfac.to_torch(split_weight_bias=split),
                M_ekfac.to_torch(split_weight_bias=split),
            )

        # now we compute the exact diagonal:
        M_ekfac.update_diag(loader)
        assert torch.norm(M_kfac.to_torch() - M_blockdiag.to_torch()) > torch.norm(
            M_ekfac.to_torch() - M_blockdiag.to_torch()
        )


@pytest.mark.filterwarnings("ignore:It is required")
def test_pspace_ekfac_vs_direct():
    """
    Check EKFAC basic operations against direct computation using
    to_torch
    """
    for get_task in [
        get_embedding_task,
        get_conv1d_task,
        get_fullyconnect_task,
        get_conv_task,
    ]:
        loader, lc, parameters, model, function = get_task()
        model.train()

        generator = TorchHooksJacobianBackend(model=model, function=function)

        M_ekfac = PMatEKFAC(generator=generator, examples=loader, layer_collection=lc)
        v = random_pvector(lc, device=device, dtype=torch.double)

        # the second time we will have called update_diag
        for i in range(2):
            M_ekfac_torch = M_ekfac.to_torch()

            vTMv_direct = torch.dot(
                torch.mv(M_ekfac_torch, v.to_torch()),
                v.to_torch(),
            )
            vTMv_ekfac = M_ekfac.vTMv(v)
            check_ratio(vTMv_direct, vTMv_ekfac)

            trace_ekfac = M_ekfac.trace()
            trace_direct = torch.trace(M_ekfac_torch)
            check_ratio(trace_direct, trace_ekfac)

            frob_ekfac = M_ekfac.norm(ord="fro")
            frob_direct = torch.linalg.matrix_norm(M_ekfac_torch)
            check_ratio(frob_direct, frob_ekfac)

            mv_direct = torch.mv(M_ekfac_torch, v.to_torch())
            mv_ekfac = M_ekfac.mv(v)
            torch.testing.assert_close(mv_direct, mv_ekfac.to_torch())

            # Test pow
            torch.testing.assert_close(
                (M_ekfac**2).to_torch(),
                torch.mm(M_ekfac_torch, M_ekfac_torch),
            )
            torch.testing.assert_close(
                torch.mm(
                    (M_ekfac ** (1 / 3)).to_torch(),
                    (M_ekfac ** (2 / 3)).to_torch(),
                ),
                M_ekfac_torch,
            )

            # Test inverse
            regul = 1e-6
            M_inv = M_ekfac.inverse(regul=regul)
            v_back = M_inv.mv(mv_ekfac + regul * v)
            torch.testing.assert_close(v.to_torch(), v_back.to_torch())

            # Test pinv
            M_inv = M_ekfac.pinv(atol=regul)
            torch.testing.assert_close(
                M_inv.mv(v).to_torch(),
                M_ekfac.solve(v, regul=regul, solve="lstsq").to_torch(),
            )

            # Test solve with vector
            v_back = M_ekfac.solve(mv_ekfac + regul * v, regul=regul)
            torch.testing.assert_close(v.to_torch(), v_back.to_torch())

            # Test solve with jacobian
            # TODO imp
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
                layer_collection=lc,
            )
            J_back = M_ekfac.solve(jaco, regul=regul)
            torch.testing.assert_close(
                stacked_v,
                J_back.to_torch(),
            )

            # Test solve lstsq with jacobian, against torch.linalg.lstsq
            # on the dense matrix
            max_eval = M_ekfac.norm(ord=2)

            torch.testing.assert_close(
                torch.linalg.lstsq(
                    M_ekfac_torch,
                    jaco.to_torch().view(-1, M_ekfac.size(0)).t(),
                    rcond=regul,
                    driver="gelsd",
                )[0],
                M_ekfac.solve(jaco, regul=regul * max_eval, solve="lstsq")
                .to_torch()
                .view(-1, M_ekfac.size(0))
                .t(),
            )

            # Test solve lstsq with vector
            v_back = M_ekfac.solve(mv_ekfac + regul * v, regul=regul)
            torch.testing.assert_close(
                torch.linalg.lstsq(
                    M_ekfac_torch,
                    mv_direct,
                    rcond=regul,
                    driver="gelsd",
                )[0],
                M_ekfac.solve(
                    mv_ekfac, regul=regul * max_eval, solve="lstsq"
                ).to_torch(),
            )

            # Test rmul
            M_mul = 1.23 * M_ekfac
            torch.testing.assert_close(1.23 * M_ekfac_torch, M_mul.to_torch())

            M_ekfac.update_diag(loader)
