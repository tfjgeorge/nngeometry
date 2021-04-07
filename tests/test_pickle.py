from tasks import get_conv_gn_task, get_conv_bn_task
from nngeometry.layercollection import LayerCollection
import pickle as pkl

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
    _, lc_different, _, _, _, _ = get_conv_bn_task()

    assert lc == lc_same
    assert lc != lc_different