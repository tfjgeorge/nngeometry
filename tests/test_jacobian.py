import torch
from tasks import get_linear_task, get_batchnorm_linear_task
from nngeometry.object.map import (DensePushForward, ImplicitPushForward,
                                   DensePullBack)
from nngeometry.generator import Jacobian
from nngeometry.object.vector import random_pvector
from utils import check_ratio, check_tensors


def update_model(model, dw):
    i = 0
    for p in model.parameters():
        j = i + p.numel()
        p.data += dw[i:j].view(*p.size())
        i = j


def get_output_vector(loader, function):
    with torch.no_grad():
        outputs = []
        for inputs, targets in loader:
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            outputs.append(function(inputs, targets))
        return torch.cat(outputs)


def test_jacobian_pushforward_dense():
    for get_task in [get_linear_task, get_batchnorm_linear_task]:
        loader, model, function = get_task()
        model.train()
        generator = Jacobian(model=model,
                             loader=loader,
                             function=function)
        push_forward = DensePushForward(generator)
        dw = random_pvector(model)

        doutput_lin = push_forward.mv(dw)

        output_before = get_output_vector(loader, function)
        update_model(model, dw.get_flat_representation())
        output_after = get_output_vector(loader, function)

        check_tensors(output_after - output_before,
                      doutput_lin.get_flat_representation())


def test_jacobian_pushforward_implicit():
    loader, model, function = get_linear_task()
    generator = Jacobian(model=model,
                         loader=loader,
                         function=function)
    dense_push_forward = DensePushForward(generator)
    implicit_push_forward = ImplicitPushForward(generator)
    dw = random_pvector(model)

    doutput_lin_dense = dense_push_forward.mv(dw)
    doutput_lin_implicit = implicit_push_forward.mv(dw)

    check_tensors(doutput_lin_dense.get_flat_representation(),
                  doutput_lin_implicit.get_flat_representation())


def test_jacobian_pullback_dense():
    loader, model, function = get_linear_task()
    generator = Jacobian(model=model,
                         loader=loader,
                         function=function)
    pull_back = DensePullBack(generator)
    push_forward = DensePushForward(generator)
    dw = random_pvector(model)

    doutput_lin = push_forward.mv(dw)
    dinput_lin = pull_back.mv(doutput_lin)
    check_ratio(torch.dot(dw.get_flat_representation(),
                          dinput_lin.get_flat_representation()),
                torch.norm(doutput_lin.get_flat_representation())**2)
