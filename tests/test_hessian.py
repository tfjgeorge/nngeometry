import math

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
from utils import check_tensors, update_model

from nngeometry import FIM, Hessian
from nngeometry.object.map import random_pfmap
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
        n_examples = len(loader.sampler)

        if isinstance(representation, PMatDense):
            torch.testing.assert_close(F.to_torch(), H.to_torch() / n_examples)

        dw = random_pvector(lc)
        torch.testing.assert_close(
            F.mv(dw).to_torch(), H.mv(dw).to_torch() / n_examples
        )

        x = random_pfmap(lc, (10, 100))
        h_mmap = H.mmap(x)
        f_mmap = F.mmap(x)
        for layer_id, layer in lc.layers.items():
            torch.testing.assert_close(
                h_mmap.to_torch_layer(layer_id)[0] / n_examples,
                f_mmap.to_torch_layer(layer_id)[0],
            )
            if layer.has_bias():
                torch.testing.assert_close(
                    h_mmap.to_torch_layer(layer_id)[1] / n_examples,
                    f_mmap.to_torch_layer(layer_id)[1],
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
        torch.testing.assert_close(
            H_dense.mv(dw).to_torch(), H_implicit.mv(dw).to_torch()
        )
        assert math.isclose(
            H_dense.vTMv(dw).item(), H_implicit.vTMv(dw).item(), abs_tol=1e-9
        )

        with pytest.raises(NotImplementedError):
            H_implicit.trace()

        x = random_pfmap(lc, (10, 3))
        dense_mmap = H_dense.mmap(x)
        imp_mmap = H_implicit.mmap(x)
        for layer_id, layer in lc.layers.items():
            torch.testing.assert_close(
                dense_mmap.to_torch_layer(layer_id)[0],
                imp_mmap.to_torch_layer(layer_id)[0],
            )
            if layer.has_bias():
                torch.testing.assert_close(
                    dense_mmap.to_torch_layer(layer_id)[1],
                    imp_mmap.to_torch_layer(layer_id)[1],
                )

        dense_solvepvec = H_dense.solve(dw, regul=1)
        for x0 in [None, dense_solvepvec]:
            torch.testing.assert_close(
                dense_solvepvec.to_torch(),
                H_implicit.solve(
                    dw, regul=1, max_iter=10000, rtol=0, atol=1e-5, x0=x0
                ).to_torch(),
                atol=1e-3,
                rtol=1e-3,
            )

        dense_solvepfmap = H_dense.solve(x, regul=1)
        for x0 in [None, dense_solvepfmap]:
            imp_solvepfmap = H_implicit.solve(
                x, regul=1, max_iter=200, rtol=0, atol=1e-5, x0=x0
            )

            for layer_id, layer in lc.layers.items():
                torch.testing.assert_close(
                    dense_solvepfmap.to_torch_layer(layer_id)[0],
                    imp_solvepfmap.to_torch_layer(layer_id)[0],
                    atol=1e-3,
                    rtol=1e-3,
                )
                if layer.has_bias():
                    torch.testing.assert_close(
                        dense_solvepfmap.to_torch_layer(layer_id)[1],
                        imp_solvepfmap.to_torch_layer(layer_id)[1],
                        atol=1e-3,
                        rtol=1e-3,
                    )


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
