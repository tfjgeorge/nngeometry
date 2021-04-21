import torch
from nngeometry.object.vector import (PVector, random_pvector,
                                      random_pvector_dict)
from nngeometry.layercollection import LayerCollection
import torch.nn as nn
import torch.nn.functional as tF
from utils import check_ratio, check_tensors
import pytest


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, 3, 1)
        self.conv2 = nn.Conv2d(5, 6, 4, 1, bias=False)
        self.conv3 = nn.Conv2d(6, 7, 3, 1)
        self.fc1 = nn.Linear(1*1*7, 10)

    def forward(self, x):
        x = tF.relu(self.conv1(x))
        x = tF.max_pool2d(x, 2, 2)
        x = tF.relu(self.conv2(x))
        x = tF.max_pool2d(x, 2, 2)
        x = tF.relu(self.conv3(x))
        x = tF.max_pool2d(x, 2, 2)
        x = x.view(-1, 1*1*7)
        x = self.fc1(x)
        return tF.log_softmax(x, dim=1)


@pytest.fixture(autouse=True)
def make_test_deterministic():
    torch.manual_seed(1234)
    yield


def test_from_dict_to_pvector():
    eps = 1e-8
    model = ConvNet()
    v = PVector.from_model(model)
    d1 = v.get_dict_representation()
    v2 = PVector(v.layer_collection, vector_repr=v.get_flat_representation())
    d2 = v2.get_dict_representation()
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
    assert torch.norm(sumr1r2.get_flat_representation() -
                      (r1.get_flat_representation() +
                       r2.get_flat_representation())) < 1e-5

    r1 = random_pvector_dict(layer_collection)
    r2 = random_pvector_dict(layer_collection)
    sumr1r2 = r1 + r2
    assert torch.norm(sumr1r2.get_flat_representation() -
                      (r1.get_flat_representation() +
                       r2.get_flat_representation())) < 1e-5

    r1 = random_pvector(layer_collection)
    r2 = random_pvector_dict(layer_collection)
    sumr1r2 = r1 + r2
    assert torch.norm(sumr1r2.get_flat_representation() -
                      (r1.get_flat_representation() +
                       r2.get_flat_representation())) < 1e-5


def test_sub():
    model = ConvNet()
    layer_collection = LayerCollection.from_model(model)
    r1 = random_pvector(layer_collection)
    r2 = random_pvector(layer_collection)
    sumr1r2 = r1 - r2
    assert torch.norm(sumr1r2.get_flat_representation() -
                      (r1.get_flat_representation() -
                       r2.get_flat_representation())) < 1e-5

    r1 = random_pvector_dict(layer_collection)
    r2 = random_pvector_dict(layer_collection)
    sumr1r2 = r1 - r2
    assert torch.norm(sumr1r2.get_flat_representation() -
                      (r1.get_flat_representation() -
                       r2.get_flat_representation())) < 1e-5

    r1 = random_pvector(layer_collection)
    r2 = random_pvector_dict(layer_collection)
    sumr1r2 = r1 - r2
    assert torch.norm(sumr1r2.get_flat_representation() -
                      (r1.get_flat_representation() -
                       r2.get_flat_representation())) < 1e-5


def test_clone():
    eps = 1e-8
    model = ConvNet()
    pvec = PVector.from_model(model)
    pvec_clone = pvec.clone()
    l_to_m, _ = pvec.layer_collection.get_layerid_module_maps(model)

    for layer_id, layer in pvec.layer_collection.layers.items():
        m = l_to_m[layer_id]
        assert m.weight is pvec.get_dict_representation()[layer_id][0]
        assert (m.weight is not
                pvec_clone.get_dict_representation()[layer_id][0])
        assert (torch.norm(m.weight -
                           pvec_clone.get_dict_representation()[layer_id][0])
                < eps)
        if m.bias is not None:
            assert m.bias is pvec.get_dict_representation()[layer_id][1]
            assert (m.bias is not
                    pvec_clone.get_dict_representation()[layer_id][1])
            assert (torch.norm(m.bias -
                               pvec_clone.get_dict_representation()[layer_id]
                               [1])
                    < eps)


def test_detach():
    eps = 1e-8
    model = ConvNet()
    pvec = PVector.from_model(model)
    pvec_clone = pvec.clone()

    # first check grad on pvec_clone
    loss = torch.norm(pvec_clone.get_flat_representation())
    loss.backward()
    pvec_clone_dict = pvec_clone.get_dict_representation()
    pvec_dict = pvec.get_dict_representation()
    for layer_id, layer in pvec.layer_collection.layers.items():
        assert torch.norm(pvec_dict[layer_id][0].grad) > eps
        assert pvec_clone_dict[layer_id][0].grad is None
        pvec_dict[layer_id][0].grad.zero_()
        if layer.bias is not None:
            assert torch.norm(pvec_dict[layer_id][1].grad) > eps
            assert pvec_clone_dict[layer_id][1].grad is None
            pvec_dict[layer_id][1].grad.zero_()

    # second check that detached grad stays at 0 when detaching
    y = torch.tensor(1., requires_grad=True)
    loss = torch.norm(pvec.detach().get_flat_representation()) + y
    loss.backward()
    for layer_id, layer in pvec.layer_collection.layers.items():
        assert torch.norm(pvec_dict[layer_id][0].grad) < eps
        if layer.bias is not None:
            assert torch.norm(pvec_dict[layer_id][1].grad) < eps


def test_norm():
    model = ConvNet()
    layer_collection = LayerCollection.from_model(model)

    v = random_pvector(layer_collection)
    check_ratio(torch.norm(v.get_flat_representation()), v.norm())

    v = random_pvector_dict(layer_collection)
    check_ratio(torch.norm(v.get_flat_representation()), v.norm())


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


def test_dot():
    model = ConvNet()
    layer_collection = LayerCollection.from_model(model)
    r1 = random_pvector(layer_collection)
    r2 = random_pvector(layer_collection)
    dotr1r2 = r1.dot(r2)
    check_ratio(torch.dot(r1.get_flat_representation(),
                          r2.get_flat_representation()),
                dotr1r2)

    r1 = random_pvector_dict(layer_collection)
    r2 = random_pvector_dict(layer_collection)
    dotr1r2 = r1.dot(r2)
    check_ratio(torch.dot(r1.get_flat_representation(),
                          r2.get_flat_representation()),
                dotr1r2)


    r1 = random_pvector(layer_collection)
    r2 = random_pvector_dict(layer_collection)
    dotr1r2 = r1.dot(r2)
    dotr2r1 = r2.dot(r1)
    check_ratio(torch.dot(r1.get_flat_representation(),
                          r2.get_flat_representation()),
                dotr1r2)
    check_ratio(torch.dot(r1.get_flat_representation(),
                          r2.get_flat_representation()),
                dotr2r1)

def test_size():
    model = ConvNet()
    layer_collection = LayerCollection.from_model(model)

    v = random_pvector(layer_collection)
    assert v.size() == v.get_flat_representation().size()

    v = random_pvector_dict(layer_collection)
    assert v.size() == v.get_flat_representation().size()