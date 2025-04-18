import torch
from tasks import get_conv_bn_task

from nngeometry.metrics import FIM
from nngeometry.object.pspace import PMatEKFACBlockDiag
from nngeometry.object.vector import random_pvector


def test_pmatmixed_ekfac():
    for get_task in [get_conv_bn_task]:
        for i in range(2):
            loader, lc, parameters, model, function = get_task()

            pmat_mixed = FIM(model=model, loader=loader, representation=PMatEKFACBlockDiag)

            dense_torch = pmat_mixed.to_torch()

            torch.testing.assert_close(torch.trace(dense_torch), pmat_mixed.trace())
            torch.testing.assert_close(torch.norm(dense_torch), pmat_mixed.frobenius_norm())

            v = random_pvector(lc)
            mv_torch = torch.mv(dense_torch, v.to_torch())
            mv_nng = pmat_mixed.mv(v)
            torch.testing.assert_close(mv_torch, mv_nng.to_torch())
            torch.testing.assert_close(
                torch.dot(v.to_torch(), torch.mv(dense_torch, v.to_torch())),
                pmat_mixed.vTMv(v),
            )

            regul = 1e-7
            v_back = pmat_mixed.solve(mv_nng + regul * v, regul=regul)
            torch.testing.assert_close(v.to_torch(), v_back.to_torch())

            if i == 0:
                pmat_mixed.update_diag(loader)
