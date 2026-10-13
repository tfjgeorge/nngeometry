import pytest
import torch
from tasks import (
    get_conv1d_task,
    get_conv_gn_task,
    get_conv_task,
    get_embedding_task,
    get_fullyconnect_task,
    get_linear_fc_task,
)

from nngeometry import Jacobian
from nngeometry.gram import GramMatrix
from nngeometry.metrics import GradientSecondMoment
from nngeometry.object.fspace import FMatDense
from nngeometry.object.map import PFMapDense, PFMapFactored
from nngeometry.object.pspace import PMatDense
from nngeometry.object.vector import random_fvector, random_pvector

nonlinear_tasks = [get_conv_gn_task, get_fullyconnect_task, get_conv_task]


def test_jacobian():
    for get_task in nonlinear_tasks:
        loader, lc, parameters, model, function = get_task()

        jacobian = Jacobian(
            model=model,
            function=function,
            loader=loader,
            layer_collection=lc,
            representation=PFMapDense,
        )
        gram = GramMatrix(
            model=model,
            function=function,
            loader=loader,
            layer_collection=lc,
            representation=FMatDense,
        )
        fim = GradientSecondMoment(
            model=model,
            function=function,
            loader=loader,
            layer_collection=lc,
            representation=PMatDense,
        )

        dw = random_pvector(layer_collection=lc)
        df = random_fvector(n_samples=jacobian.size(1), n_output=jacobian.size(0))

        torch.testing.assert_close(
            gram.to_torch(), (jacobian @ jacobian.adjoint()).to_torch()
        )
        torch.testing.assert_close(
            fim.to_torch(),
            (1 / jacobian.size(1) * (jacobian.adjoint() @ jacobian)).to_torch(),
        )
        torch.testing.assert_close(
            jacobian.to_torch(), jacobian.adjoint().adjoint().to_torch()
        )

        torch.testing.assert_close(
            (jacobian @ dw).to_torch(), (dw @ jacobian.adjoint()).to_torch()
        )
        torch.testing.assert_close(
            (df @ jacobian).to_torch(), (jacobian.adjoint() @ df).to_torch()
        )

        with pytest.raises(TypeError):
            jacobian.adjoint() @ jacobian.adjoint()
        with pytest.raises(TypeError):
            jacobian @ jacobian
        with pytest.raises(TypeError):
            jacobian.adjoint() @ dw
        with pytest.raises(TypeError):
            df @ jacobian.adjoint()


def test_jacobian_factored():
    for get_task in [
        get_fullyconnect_task,
        get_conv_task,
        get_conv1d_task,
        get_embedding_task,
        get_linear_fc_task,
    ]:
        loader, lc, parameters, model, function = get_task()

        jacobian = Jacobian(
            model=model,
            function=function,
            loader=loader,
            layer_collection=lc,
            representation=PFMapDense,
        )
        jacobian_factored = Jacobian(
            model=model,
            function=function,
            loader=loader,
            layer_collection=lc,
            representation=PFMapFactored,
        )

        jacobian_factored.size() == jacobian.size()

        for layer_id in lc.layers:
            for w_dense, w_factored in zip(
                jacobian.to_torch_layer(layer_id),
                jacobian_factored.to_torch_layer(layer_id),
            ):
                torch.testing.assert_close(w_dense, w_factored)

        torch.testing.assert_close(jacobian.to_torch(), jacobian_factored.to_torch())


def test_jacobian_factored_rejects_unsupported_layers():
    loader, lc, parameters, model, function = get_conv_gn_task()

    with pytest.raises(NotImplementedError):
        Jacobian(
            model=model,
            function=function,
            loader=loader,
            layer_collection=lc,
            representation=PFMapFactored,
        )
