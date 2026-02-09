import pytest
import torch
import torch.nn as nn
import torch.nn.functional as tF
from tasks import ConvNet
from utils import check_ratio, check_tensors

from nngeometry.layercollection import LayerCollection
from nngeometry.object.vector import PVector, random_pvector, random_pvector_dict


@pytest.fixture(autouse=True)
def make_test_deterministic():
    torch.manual_seed(1234)
    yield


def test_from_dict_to_pvector():
    eps = 1e-8
    model = ConvNet()
    v = PVector.from_model(model)
    d1 = v.to_dict()
    v2 = PVector(v.layer_collection, vector_repr=v.to_torch())
    d2 = v2.to_dict()
    assert d1.keys() == d2.keys()
    for k in d1.keys():
        assert torch.norm(d1[k][0] - d2[k][0]) < eps
        if len(d1[k]) == 2:
            assert torch.norm(d1[k][1] - d2[k][1]) < eps


def test_add():
    model = ConvNet()
    layer_collection = LayerCollection.from_model(model)
    r1 = random_pvector(layer_collection)
    r2 = random_pvector(layer_collection)
    sumr1r2 = r1 + r2
    assert torch.norm(sumr1r2.to_torch() - (r1.to_torch() + r2.to_torch())) < 1e-5

    r1 = random_pvector_dict(layer_collection)
    r2 = random_pvector_dict(layer_collection)
    sumr1r2 = r1 + r2
    assert torch.norm(sumr1r2.to_torch() - (r1.to_torch() + r2.to_torch())) < 1e-5

    r1 = random_pvector(layer_collection)
    r2 = random_pvector_dict(layer_collection)
    sumr1r2 = r1 + r2
    assert torch.norm(sumr1r2.to_torch() - (r1.to_torch() + r2.to_torch())) < 1e-5


def test_sub():
    model = ConvNet()
    layer_collection = LayerCollection.from_model(model)
    r1 = random_pvector(layer_collection)
    r2 = random_pvector(layer_collection)
    sumr1r2 = r1 - r2
    assert torch.norm(sumr1r2.to_torch() - (r1.to_torch() - r2.to_torch())) < 1e-5

    r1 = random_pvector_dict(layer_collection)
    r2 = random_pvector_dict(layer_collection)
    sumr1r2 = r1 - r2
    assert torch.norm(sumr1r2.to_torch() - (r1.to_torch() - r2.to_torch())) < 1e-5

    r1 = random_pvector(layer_collection)
    r2 = random_pvector_dict(layer_collection)
    sumr1r2 = r1 - r2
    assert torch.norm(sumr1r2.to_torch() - (r1.to_torch() - r2.to_torch())) < 1e-5


def test_pow():
    model = ConvNet()
    layer_collection = LayerCollection.from_model(model)
    r1 = random_pvector(layer_collection)
    sqrt_r1 = r1**3
    assert torch.norm(sqrt_r1.to_torch() - r1.to_torch() ** 3) < 1e-5

    r1 = random_pvector_dict(layer_collection)
    sqrt_r1 = r1**3
    assert torch.norm(sqrt_r1.to_torch() - r1.to_torch() ** 3) < 1e-5


def test_clone():
    eps = 1e-8
    model = ConvNet()
    pvec = PVector.from_model(model)
    pvec_clone = pvec.clone()
    l_to_m = pvec.layer_collection.get_layerid_module_map(model)

    for layer_id, layer in pvec.layer_collection.layers.items():
        m = l_to_m[layer_id]
        assert m.weight is pvec.to_dict()[layer_id][0]
        assert m.weight is not pvec_clone.to_dict()[layer_id][0]
        assert torch.norm(m.weight - pvec_clone.to_dict()[layer_id][0]) < eps
        if layer.has_bias():
            assert m.bias is pvec.to_dict()[layer_id][1]
            assert m.bias is not pvec_clone.to_dict()[layer_id][1]
            assert torch.norm(m.bias - pvec_clone.to_dict()[layer_id][1]) < eps


def test_detach():
    eps = 1e-8
    model = ConvNet()
    pvec = PVector.from_model(model)
    pvec_clone = pvec.clone()

    # first check grad on pvec_clone
    loss = torch.norm(pvec_clone.to_torch())
    loss.backward()
    pvec_clone_dict = pvec_clone.to_dict()
    pvec_dict = pvec.to_dict()
    for layer_id, layer in pvec.layer_collection.layers.items():
        assert torch.norm(pvec_dict[layer_id][0].grad) > eps
        assert pvec_clone_dict[layer_id][0].grad is None
        pvec_dict[layer_id][0].grad.zero_()
        if layer.has_bias():
            assert torch.norm(pvec_dict[layer_id][1].grad) > eps
            assert pvec_clone_dict[layer_id][1].grad is None
            pvec_dict[layer_id][1].grad.zero_()

    # second check that detached grad stays at 0 when detaching
    y = torch.tensor(1.0, requires_grad=True)
    loss = torch.norm(pvec.detach().to_torch()) + y
    loss.backward()
    for layer_id, layer in pvec.layer_collection.layers.items():
        assert torch.norm(pvec_dict[layer_id][0].grad) < eps
        if layer.has_bias():
            assert torch.norm(pvec_dict[layer_id][1].grad) < eps


def test_norm():
    model = ConvNet()
    layer_collection = LayerCollection.from_model(model)

    v = random_pvector(layer_collection)
    check_ratio(torch.norm(v.to_torch()), v.norm())

    v = random_pvector_dict(layer_collection)
    check_ratio(torch.norm(v.to_torch()), v.norm())


def test_from_to_model():
    model1 = ConvNet()
    model2 = ConvNet()

    w1 = PVector.from_model(model1).clone()
    w2 = PVector.from_model(model2).clone()

    model3 = ConvNet()
    w1.copy_to_model(model3)
    # now model1 and model3 should be the same

    for p1, p3 in zip(model1.parameters(), model3.parameters()):
        check_tensors(p1, p3)

    ###
    diff_1_2 = w2 - w1
    diff_1_2.add_to_model(model3)
    # now model2 and model3 should be the same

    for p2, p3 in zip(model2.parameters(), model3.parameters()):
        check_tensors(p2, p3)


def test_from_to_model_with_lc():
    model1 = ConvNet()
    model2 = ConvNet()

    lc = LayerCollection()
    lc_other = LayerCollection()
    for layer_name, mod in model1.named_modules():
        if len(list(mod.children())) == 0:
            if layer_name != "conv3":
                lc.add_layer_from_model(model1, mod)
            else:
                lc_other.add_layer_from_model(model1, mod)

    w1 = PVector.from_model(model1, layer_collection=lc).clone()
    w2 = PVector.from_model(model2, layer_collection=lc).clone()

    model3 = ConvNet()
    w1.copy_to_model(model3)
    # now model1 and model3 should be the same

    for id1, p1 in model1.named_parameters():
        p3 = model3.get_parameter(id1)
        if id1.split(".")[0] == "conv3":
            assert torch.norm(p1 - p3) > 0.1
        else:
            torch.testing.assert_close(p1, p3)

    ###
    diff_1_2 = w2 - w1
    diff_1_2.add_to_model(model3)
    # now model2 and model3 should be the same

    for id2, p2 in model2.named_parameters():
        p3 = model3.get_parameter(id2)
        if id2.split(".")[0] == "conv3":
            assert torch.norm(p2 - p3) > 0.1
        else:
            torch.testing.assert_close(p2, p3)


def test_dot():
    model = ConvNet()
    layer_collection = LayerCollection.from_model(model)
    r1 = random_pvector(layer_collection)
    r2 = random_pvector(layer_collection)
    dotr1r2 = r1.dot(r2)
    check_ratio(torch.dot(r1.to_torch(), r2.to_torch()), dotr1r2)

    r1 = random_pvector_dict(layer_collection)
    r2 = random_pvector_dict(layer_collection)
    dotr1r2 = r1.dot(r2)
    check_ratio(torch.dot(r1.to_torch(), r2.to_torch()), dotr1r2)

    r1 = random_pvector(layer_collection)
    r2 = random_pvector_dict(layer_collection)
    dotr1r2 = r1.dot(r2)
    dotr2r1 = r2.dot(r1)
    check_ratio(torch.dot(r1.to_torch(), r2.to_torch()), dotr1r2)
    check_ratio(torch.dot(r1.to_torch(), r2.to_torch()), dotr2r1)

    r1 = random_pvector(layer_collection)
    r2 = random_pvector_dict(layer_collection)
    dotr1r2 = r1 @ r2
    dotr2r1 = r2 @ r1
    check_ratio(r1.to_torch() @ r2.to_torch(), dotr1r2)
    check_ratio(r1.to_torch() @ r2.to_torch(), dotr2r1)


def test_inplace():
    model = ConvNet()
    layer_collection = LayerCollection.from_model(model)
    for r1, r2 in [
        (random_pvector(layer_collection), random_pvector(layer_collection)),
        (random_pvector_dict(layer_collection), random_pvector_dict(layer_collection)),
    ]:
        og_r1 = r1.clone()
        iopr1r2 = og_r1
        iopr1r2 += r2
        opr1r2 = r1 + r2
        check_tensors(opr1r2.to_torch(), iopr1r2.to_torch())
        assert not torch.allclose(og_r1.to_torch(), r1.to_torch())

        og_r1 = r1.clone()
        iopr1r2 = og_r1
        iopr1r2 -= r2
        opr1r2 = r1 - r2
        check_tensors(opr1r2.to_torch(), iopr1r2.to_torch())
        assert not torch.allclose(og_r1.to_torch(), r1.to_torch())

        og_r1 = r1.clone()
        iopr1r2 = og_r1
        iopr1r2 *= 2
        opr1r2 = 2 * r1
        check_tensors(opr1r2.to_torch(), iopr1r2.to_torch())
        assert not torch.allclose(og_r1.to_torch(), r1.to_torch())

    # can't use inplace operations for different representations
    r1, r2 = random_pvector(layer_collection), random_pvector_dict(layer_collection)
    with pytest.raises(NotImplementedError):
        r1 += r2
    with pytest.raises(NotImplementedError):
        r1 -= r2


def test_size():
    model = ConvNet()
    layer_collection = LayerCollection.from_model(model)

    v = random_pvector(layer_collection)
    assert v.size() == v.to_torch().size()

    v = random_pvector_dict(layer_collection)
    assert v.size() == v.to_torch().size()
