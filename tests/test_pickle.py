from tasks import get_conv_gn_task, get_conv_task
from nngeometry.layercollection import LayerCollection
import pickle as pkl
from utils import check_tensors

from nngeometry.object.pspace import (PMatDense, PMatDiag, PMatBlockDiag,
                                      PMatLowRank, PMatQuasiDiag)
from nngeometry.generator import Jacobian

def test_layercollection_pkl():
    _, lc, _, _, _, _ = get_conv_gn_task()

    with open('/tmp/lc.pkl', 'wb') as f:
        pkl.dump(lc, f)

    with open('/tmp/lc.pkl', 'rb') as f:
        lc_pkl = pkl.load(f)

    assert lc == lc_pkl


def test_layercollection_eq():
    _, lc, _, _, _, _ = get_conv_gn_task()
    _, lc_same, _, _, _, _ = get_conv_gn_task()
    _, lc_different, _, _, _, _ = get_conv_task()

    assert lc == lc_same
    assert lc != lc_different


def test_PMat_pickle():
    loader, lc, parameters, model, function, n_output = get_conv_task()

    generator = Jacobian(layer_collection=lc,
                         model=model,
                         function=function,
                         n_output=n_output)
                    
    for repr in [PMatDense, PMatDiag, PMatBlockDiag,
                 PMatLowRank, PMatQuasiDiag]:
        PMat = repr(generator=generator,
                    examples=loader)

        with open('/tmp/PMat.pkl', 'wb') as f:
            pkl.dump(PMat, f)

        with open('/tmp/PMat.pkl', 'rb') as f:
            PMat_pkl = pkl.load(f)

        check_tensors(PMat.get_dense_tensor(), PMat_pkl.get_dense_tensor())