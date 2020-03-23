import torch
from tasks import (get_linear_task, get_batchnorm_linear_task,
                   get_fullyconnect_onlylast_task,
                   get_fullyconnect_task)
from nngeometry.object.map import (PushForwardDense, PushForwardImplicit,
                                   PullBackDense)
from nngeometry.object.fspace import FSpaceDense
from nngeometry.generator import Jacobian
from nngeometry.object.vector import random_pvector
from utils import check_ratio, check_tensors


linear_tasks = [get_linear_task, get_batchnorm_linear_task,
                get_fullyconnect_onlylast_task]

nonlinear_tasks = [get_fullyconnect_task]


def update_model(parameters, dw):
    i = 0
    for p in parameters:
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


def test_jacobian_pushforward_dense_linear():
    for get_task in linear_tasks:
        loader, lc, parameters, model, function, n_output = get_task()
        model.train()
        generator = Jacobian(layer_collection=lc,
                             model=model,
                             loader=loader,
                             function=function,
                             n_output=n_output)
        push_forward = PushForwardDense(generator)
        dw = random_pvector(lc, device='cuda')

        doutput_lin = push_forward.mv(dw)

        output_before = get_output_vector(loader, function)
        update_model(parameters, dw.get_flat_representation())
        output_after = get_output_vector(loader, function)

        check_tensors(output_after - output_before,
                      doutput_lin.get_flat_representation().t())


def test_jacobian_pushforward_dense_nonlinear():
    for get_task in nonlinear_tasks:
        loader, lc, parameters, model, function, n_output = get_task()
        model.train()
        generator = Jacobian(layer_collection=lc,
                             model=model,
                             loader=loader,
                             function=function,
                             n_output=n_output)
        push_forward = PushForwardDense(generator)
        dw = 1e-4 * random_pvector(lc, device='cuda')

        doutput_lin = push_forward.mv(dw)

        output_before = get_output_vector(loader, function)
        update_model(parameters, dw.get_flat_representation())
        output_after = get_output_vector(loader, function)

        check_tensors(output_after - output_before,
                      doutput_lin.get_flat_representation().t(),
                      1e-2)


def test_jacobian_pushforward_implicit():
    for get_task in linear_tasks:
        loader, lc, parameters, model, function, n_output = get_task()
        generator = Jacobian(layer_collection=lc,
                             model=model,
                             loader=loader,
                             function=function,
                             n_output=n_output)
        dense_push_forward = PushForwardDense(generator)
        implicit_push_forward = PushForwardImplicit(generator)
        dw = random_pvector(lc, device='cuda')

        doutput_lin_dense = dense_push_forward.mv(dw)
        doutput_lin_implicit = implicit_push_forward.mv(dw)

        check_tensors(doutput_lin_dense.get_flat_representation(),
                      doutput_lin_implicit.get_flat_representation())


def test_jacobian_pullback_dense():
    for get_task in linear_tasks:
        loader, lc, parameters, model, function, n_output = get_task()
        generator = Jacobian(layer_collection=lc,
                             model=model,
                             loader=loader,
                             function=function,
                             n_output=n_output)
        pull_back = PullBackDense(generator)
        push_forward = PushForwardDense(generator)
        dw = random_pvector(lc, device='cuda')

        doutput_lin = push_forward.mv(dw)
        dinput_lin = pull_back.mv(doutput_lin)
        check_ratio(torch.dot(dw.get_flat_representation(),
                              dinput_lin.get_flat_representation()),
                    torch.norm(doutput_lin.get_flat_representation())**2)


def test_jacobian_fdense_vs_pushforward():
    for get_task in linear_tasks + nonlinear_tasks:
        loader, lc, parameters, model, function, n_output = get_task()
        model.train()
        generator = Jacobian(layer_collection=lc,
                             model=model,
                             loader=loader,
                             function=function,
                             n_output=n_output)
        push_forward = PushForwardDense(generator)
        fspace_dense = FSpaceDense(generator)
        # dw = random_pvector(lc, device='cuda')

        jacobian = push_forward.get_tensor()
        sj = jacobian.size()
        fspace_computed = torch.mm(jacobian.view(-1, sj[2]),
                                   jacobian.view(-1, sj[2]).t())
        check_tensors(fspace_computed.view(sj[0], sj[1], sj[0], sj[1]),
                      fspace_dense.get_tensor())
