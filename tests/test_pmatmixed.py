from copy import deepcopy

import torch
from tasks import get_conv_bn_task, device

from nngeometry.layercollection import LinearLayer
from nngeometry.metrics import FIM
from nngeometry.object.map import PFMapDense, random_pfmap
from nngeometry.object.pspace import (
    PMatBlockDiag,
    PMatDense,
    PMatEKFACBlockDiag,
    PMatMixed,
)
from nngeometry.object.vector import random_pvector


def test_pmatmixed_ekfac():
    for get_task in [get_conv_bn_task]:
        for i in range(2):
            loader, lc, parameters, model, function = get_task()

            pmat_mixed = FIM(
                model=model, loader=loader, representation=PMatEKFACBlockDiag
            )

            dense_torch = pmat_mixed.to_torch()

            torch.testing.assert_close(torch.trace(dense_torch), pmat_mixed.trace())
            torch.testing.assert_close(
                torch.norm(dense_torch), pmat_mixed.frobenius_norm()
            )

            x = 2
            torch.testing.assert_close((x * pmat_mixed).to_torch(), x * dense_torch)

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

            # Test inverse
            pmat_mixed_inv = pmat_mixed.inverse(regul=regul)
            v_back = pmat_mixed_inv.mv(mv_nng + regul * v)
            torch.testing.assert_close(v.to_torch(), v_back.to_torch())

            # Test solve with jacobian
            c = 1.678
            stacked_mv = torch.stack([c**i * mv_torch for i in range(6)]).reshape(
                2, 3, -1
            )
            stacked_v = torch.stack([c**i * v.to_torch() for i in range(6)]).reshape(
                2, 3, -1
            )
            jaco = PFMapDense(
                generator=pmat_mixed.generator,
                data=stacked_mv + regul * stacked_v,
                layer_collection=lc,
            )
            J_back = pmat_mixed.solve(jaco, regul=regul)
            torch.testing.assert_close(
                stacked_v,
                J_back.to_torch(),
            )

            pfmap = random_pfmap(lc, output_size=(3, 4), device=device)

            mapTMmap_direct = torch.zeros((4,))
            pfmap_torch = pfmap.to_torch()
            for i in range(4):
                for j in range(3):
                    mapTMmap_direct[i] += torch.dot(
                        torch.mv(dense_torch, pfmap_torch[j, i]),
                        pfmap_torch[j, i],
                    )
            mapTMmap_ekfac = pmat_mixed.mapTMmap(pfmap, reduction="sum")
            torch.testing.assert_close(mapTMmap_direct, mapTMmap_ekfac)

            mapTMmap_direct = torch.zeros((3, 4))
            pfmap_torch = pfmap.to_torch()
            for i in range(4):
                for j in range(3):
                    mapTMmap_direct[j, i] += torch.dot(
                        torch.mv(dense_torch, pfmap_torch[j, i]),
                        pfmap_torch[j, i],
                    )
            mapTMmap_ekfac = pmat_mixed.mapTMmap(pfmap, reduction="diag")
            torch.testing.assert_close(mapTMmap_direct, mapTMmap_ekfac)

            # 2nd time the diag is updated
            if i == 0:
                pmat_mixed.update_diag(loader)


class PMatMixedNoUpdateDiag(PMatMixed):
    """A mixed representation only used for test below"""

    def __init__(self, layer_collection, generator, data=None, examples=None, **kwargs):
        pmm = PMatMixed.from_mapping(
            layer_collection,
            generator,
            data=data,
            examples=examples,
            default_representation=PMatDense,
            map_layers_to={
                LinearLayer: PMatBlockDiag,
            },
            **kwargs,
        )
        super().__init__(
            layer_collection,
            generator,
            pmm.layer_collection_each,
            pmm.layer_map,
            pmm.sub_pmats,
        )


def test_pmat_mixed_no_ekfac():
    for get_task in [get_conv_bn_task]:
        loader, lc, parameters, model, function = get_task()
        pmat_custom = FIM(
            model=model, loader=loader, representation=PMatMixedNoUpdateDiag
        )
        assert not hasattr(pmat_custom, "update_diag")

        pmat_ekfac_bd = FIM(
            model=model, loader=loader, representation=PMatEKFACBlockDiag
        )
        assert hasattr(pmat_ekfac_bd, "update_diag")
        cp = deepcopy(pmat_ekfac_bd)
        # can't update_diag because of dummy generator
        assert not hasattr(cp, "update_diag")
