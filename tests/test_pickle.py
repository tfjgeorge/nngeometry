import pickle as pkl

from tasks import get_conv_gn_task, get_conv_task
from utils import check_tensors

from nngeometry.backend import TorchHooksJacobianBackend
from nngeometry.object.pspace import (
    PMatBlockDiag,
    PMatDense,
    PMatDiag,
    PMatLowRank,
    PMatQuasiDiag,
)
from nngeometry.object.vector import PVector


def test_layercollection_pkl():
    _, lc, _, _, _ = get_conv_gn_task()

    with open("/tmp/lc.pkl", "wb") as f:
        pkl.dump(lc, f)

    with open("/tmp/lc.pkl", "rb") as f:
        lc_pkl = pkl.load(f)

    assert lc == lc_pkl


def test_layercollection_eq():
    _, lc, _, _, _ = get_conv_gn_task()
    _, lc_same, _, _, _ = get_conv_gn_task()
    _, lc_different, _, _, _ = get_conv_task()

    assert lc == lc_same
    assert lc != lc_different


def test_PMat_pickle():
    loader, lc, parameters, model, function = get_conv_task()

    generator = TorchHooksJacobianBackend(
        layer_collection=lc,
        model=model,
        function=function,
    )

    for repr in [PMatDense, PMatDiag, PMatBlockDiag, PMatLowRank, PMatQuasiDiag]:
        PMat = repr(generator=generator, examples=loader)

        with open("/tmp/PMat.pkl", "wb") as f:
            pkl.dump(PMat, f)

        with open("/tmp/PMat.pkl", "rb") as f:
            PMat_pkl = pkl.load(f)

        check_tensors(PMat.get_dense_tensor(), PMat_pkl.get_dense_tensor())


def test_PVector_pickle():
    _, _, _, model, _ = get_conv_task()

    vec = PVector.from_model(model)

    with open("/tmp/PVec.pkl", "wb") as f:
        pkl.dump(vec, f)

    with open("/tmp/PVec.pkl", "rb") as f:
        vec_pkl = pkl.load(f)

    check_tensors(vec.get_flat_representation(), vec_pkl.get_flat_representation())
