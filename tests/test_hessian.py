import pytest
import torch
from tasks import (
    device,
    get_conv_task,
    get_fullyconnect_onlylast_task,
    get_fullyconnect_task,
    get_linear_conv_task,
    get_linear_fc_task,
    to_device,
)
from utils import check_ratio, check_tensors, update_model

from nngeometry import FIM, Hessian
from nngeometry.object.pspace import PMatDense, PMatImplicit
from nngeometry.object.vector import PVector, random_pvector
from nngeometry.utils import grad

linear_tasks = [
    get_linear_fc_task,
    get_linear_conv_task,
    get_fullyconnect_onlylast_task,
]
nonlinear_tasks = [get_fullyconnect_task, get_conv_task]


@pytest.fixture(autouse=True)
def make_test_deterministic():
    torch.manual_seed(1234)
    yield


@pytest.mark.parametrize("representation", [PMatDense, PMatImplicit])
def test_hessian_vs_FIM(representation):
    for get_task in linear_tasks:
        loader, lc, parameters, model, _ = get_task()
        model.train()

        F = FIM(
            layer_collection=lc,
            model=model,
            loader=loader,
            variant="classif_logits",
            representation=representation,
            function=lambda *d: model(to_device(d[0])),
        )

        def f(y_pred, y):
            return torch.nn.functional.cross_entropy(y_pred, y, reduction="sum")

        H = Hessian(
            layer_collection=lc,
            model=model,
            loader=loader,
            representation=representation,
            function=f,
        )

        if isinstance(representation, PMatDense):
            torch.testing.assert_close(F.to_torch(), H.to_torch() / len(loader.sampler))
        else:
            dw = random_pvector(lc)
            torch.testing.assert_close(
                F.mv(dw).to_torch(), H.mv(dw).to_torch() / len(loader.sampler)
            )


def test_Hdense_vs_Himplicit():
    for get_task in linear_tasks + nonlinear_tasks:
        loader, lc, parameters, model, function = get_task()
        model.train()

        def f(y_pred, y):
            return torch.nn.functional.cross_entropy(y_pred, y, reduction="sum")

        H_dense = Hessian(
            layer_collection=lc,
            model=model,
            loader=loader,
            representation=PMatDense,
            function=f,
        )

        H_implicit = Hessian(
            layer_collection=lc,
            model=model,
            loader=loader,
            representation=PMatImplicit,
            function=f,
        )

        dw = random_pvector(lc)
        check_tensors(H_dense.mv(dw).to_torch(), H_implicit.mv(dw).to_torch())
        check_ratio(H_dense.vTMv(dw), H_implicit.vTMv(dw))
        check_ratio(H_dense.trace(), H_implicit.trace(500), eps=0.1)


def test_H_vs_linearization():
    step = 1e-5

    for get_task in nonlinear_tasks:
        loader, lc, parameters, model, function = get_task()
        model.train()

        def f(y_pred, y):
            return torch.nn.functional.cross_entropy(y_pred, y, reduction="sum")

        H = Hessian(
            layer_collection=lc,
            model=model,
            loader=loader,
            representation=PMatDense,
            function=f,
        )

        X, y = loader.dataset.tensors
        loss = torch.nn.functional.cross_entropy(model(X), y, reduction="sum")

        params = PVector.from_model(model=model)

        grad_before = grad(loss, params)

        dw = random_pvector(lc, device=device)
        dw = step / dw.norm() * dw

        update_model(parameters, dw.to_torch())

        loss = torch.nn.functional.cross_entropy(model(X), y, reduction="sum")
        grad_after = grad(loss, params)

        delta = H.mv(dw)

        check_tensors(
            (grad_after - grad_before).to_torch(),
            delta.to_torch(),
        )
