import torch
from nngeometry.vector import (PVector, random_pvector,
                               random_pvector_dict)
import torch.nn as nn
import torch.nn.functional as tF
from nngeometry.utils import get_individual_modules


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, 3, 1)
        self.conv2 = nn.Conv2d(5, 6, 4, 1)
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


def test_from_dict_to_pvector():
    eps = 1e-8
    model = ConvNet()
    v = PVector.from_model(model)
    d1 = v.get_dict_representation()
    v2 = PVector(model, vector_repr=v.get_flat_representation())
    d2 = v2.get_dict_representation()
    assert d1.keys() == d2.keys()
    for k in d1.keys():
        assert torch.norm(d1[k][0] - d2[k][0]) < eps
        if len(d1[k]) == 2:
            assert torch.norm(d1[k][1] - d2[k][1]) < eps


def test_add():
    model = ConvNet()
    r1 = random_pvector(model)
    r2 = random_pvector(model)
    sumr1r2 = r1 + r2
    assert torch.norm(sumr1r2.get_flat_representation() -
                      (r1.get_flat_representation() +
                       r2.get_flat_representation())) < 1e-5

    r1 = random_pvector_dict(model)
    r2 = random_pvector_dict(model)
    sumr1r2 = r1 + r2
    assert torch.norm(sumr1r2.get_flat_representation() -
                      (r1.get_flat_representation() +
                       r2.get_flat_representation())) < 1e-5

    r1 = random_pvector(model)
    r2 = random_pvector_dict(model)
    sumr1r2 = r1 + r2
    assert torch.norm(sumr1r2.get_flat_representation() -
                      (r1.get_flat_representation() +
                       r2.get_flat_representation())) < 1e-5


def test_sub():
    model = ConvNet()
    r1 = random_pvector(model)
    r2 = random_pvector(model)
    sumr1r2 = r1 - r2
    assert torch.norm(sumr1r2.get_flat_representation() -
                      (r1.get_flat_representation() -
                       r2.get_flat_representation())) < 1e-5

    r1 = random_pvector_dict(model)
    r2 = random_pvector_dict(model)
    sumr1r2 = r1 - r2
    assert torch.norm(sumr1r2.get_flat_representation() -
                      (r1.get_flat_representation() -
                       r2.get_flat_representation())) < 1e-5

    r1 = random_pvector(model)
    r2 = random_pvector_dict(model)
    sumr1r2 = r1 - r2
    assert torch.norm(sumr1r2.get_flat_representation() -
                      (r1.get_flat_representation() -
                       r2.get_flat_representation())) < 1e-5


def test_clone():
    eps = 1e-8
    model = ConvNet()
    pvec = PVector.from_model(model)
    pvec_clone = pvec.clone()
    mods, p_pos = get_individual_modules(model)
    for m in mods:
        assert m.weight is pvec.get_dict_representation()[m][0]
        assert m.weight is not pvec_clone.get_dict_representation()[m][0]
        assert torch.norm(m.weight -
                          pvec_clone.get_dict_representation()[m][0]) < eps
        if m.bias is not None:
            assert m.bias is pvec.get_dict_representation()[m][1]
            assert m.bias is not pvec_clone.get_dict_representation()[m][1]
            assert torch.norm(m.bias -
                              pvec_clone.get_dict_representation()[m][1]) < eps


def test_detach():
    eps = 1e-8
    model = ConvNet()
    pvec = PVector.from_model(model)
    pvec_clone = pvec.clone()
    mods, p_pos = get_individual_modules(model)

    # first check grad on pvec_clone
    loss = torch.norm(pvec_clone.get_flat_representation())
    loss.backward()
    pvec_clone_dict = pvec_clone.get_dict_representation()
    pvec_dict = pvec.get_dict_representation()
    for m in mods:
        assert torch.norm(pvec_dict[m][0].grad) > eps
        assert pvec_clone_dict[m][0].grad is None
        pvec_dict[m][0].grad.zero_()
        if m.bias is not None:
            assert torch.norm(pvec_dict[m][1].grad) > eps
            assert pvec_clone_dict[m][1].grad is None
            pvec_dict[m][1].grad.zero_()

    # second check that detached grad stays at 0 when detaching
    y = torch.tensor(1., requires_grad=True)
    loss = torch.norm(pvec.detach().get_flat_representation()) + y
    loss.backward()
    for m in mods:
        assert torch.norm(pvec_dict[m][0].grad) < eps
        if m.bias is not None:
            assert torch.norm(pvec_dict[m][1].grad) < eps
