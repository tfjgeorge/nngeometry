import torch
from nngeometry.object.vector import PVector, random_pvector
from nngeometry.utils import grad
from utils import check_tensors
import pytest
from tasks import get_conv_gn_task, to_device

@pytest.fixture(autouse=True)
def make_test_deterministic():
    torch.manual_seed(1234)
    yield


def test_grad_dict_repr():
    loader, lc, parameters, model, function, n_output = get_conv_gn_task()

    d, _ = next(iter(loader))
    scalar_output = model(to_device(d)).sum()
    vec = PVector.from_model(model)

    grad_nng = grad(scalar_output, vec, retain_graph=True)

    scalar_output.backward()
    grad_direct = PVector.from_model_grad(model)

    check_tensors(grad_direct.get_flat_representation(),
                  grad_nng.get_flat_representation())


def test_grad_flat_repr():
    loader, lc, parameters, model, function, n_output = get_conv_gn_task()

    vec = random_pvector(lc)
    scalar_output = vec.norm()

    with pytest.raises(RuntimeError):
        grad(scalar_output, vec)
