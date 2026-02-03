import pytest
import torch
from tasks import get_conv_bn_task

from nngeometry.metrics import FIM
from nngeometry.object.map import PFMapDense
from nngeometry.object.pspace import PMatEye
from nngeometry.object.vector import random_pvector


def test_pmateye():
    get_task = get_conv_bn_task

    loader, lc, parameters, model, function = get_task()

    pmat_eye = PMatEye(layer_collection=lc)

    for i in range(3):
        dense_torch = pmat_eye.to_torch()

        torch.testing.assert_close(torch.trace(dense_torch), pmat_eye.trace())
        for ord in [-2, 2, "fro"]:
            torch.testing.assert_close(
                torch.linalg.norm(dense_torch, ord=ord),
                pmat_eye.norm(ord),
                atol=1e-4,
                rtol=1e-4,
            )
        with pytest.raises(RuntimeError):
            pmat_eye.norm("prout")

        v = random_pvector(lc, dtype=torch.float32)
        mv_torch = torch.mv(dense_torch, v.to_torch())
        mv_nng = pmat_eye.mv(v)
        torch.testing.assert_close(mv_torch, mv_nng.to_torch())
        torch.testing.assert_close(
            torch.dot(v.to_torch(), torch.mv(dense_torch, v.to_torch())),
            pmat_eye.vTMv(v),
        )

        regul = 1e-7
        v_back = pmat_eye.solve(mv_nng + regul * v, regul=regul)
        torch.testing.assert_close(v.to_torch(), v_back.to_torch())

        # Test solve with jacobian
        c = 1.678
        stacked_mv = torch.stack([c**i * mv_torch for i in range(6)]).reshape(2, 3, -1)
        stacked_v = torch.stack([c**i * v.to_torch() for i in range(6)]).reshape(
            2, 3, -1
        )
        jaco = PFMapDense(
            generator=None,
            data=stacked_mv + regul * stacked_v,
            layer_collection=lc,
        )
        J_back = pmat_eye.solve(jaco, regul=regul)
        torch.testing.assert_close(
            stacked_v,
            J_back.to_torch(),
        )

        pmat_eye2 = -1.234 * pmat_eye
        assert torch.norm(pmat_eye2.to_torch() - pmat_eye.to_torch()) > 0.1
        pmat_eye = pmat_eye2

        # test inverse
        regul = 1e-5
        torch.testing.assert_close(
            torch.eye(lc.numel(), dtype=torch.float32),
            torch.mm(
                pmat_eye2.to_torch()
                + regul * torch.eye(lc.numel(), dtype=torch.float32),
                pmat_eye2.inverse(regul=regul).to_torch(),
            ),
        )

        # test pinv
        torch.testing.assert_close(
            torch.zeros((lc.numel(), lc.numel()), dtype=torch.float64),
            pmat_eye2.pinv(atol=10).to_torch(),
        )
