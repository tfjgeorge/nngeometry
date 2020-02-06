import torch
from tasks import get_linear_task
from nngeometry.object.map import DensePushForward, ImplicitPushForward
from nngeometry.generator import Jacobian
from nngeometry.object.vector import random_pvector
from utils import check_ratio, check_tensors


def update_model(model, dw):
    i = 0
    for p in model.parameters():
        j = i + p.numel()
        p.data += dw[i:j].view(*p.size())
        i = j


def get_output_vector(loader, output_fn):
    with torch.no_grad():
        outputs = []
        for inputs, targets in loader:
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            outputs.append(output_fn(inputs, targets))
        return torch.cat(outputs)


def test_jacobian_pushforward_dense():
    loader, model, output_fn = get_linear_task()
    generator = Jacobian(model=model,
                         loader=loader,
                         output_fn=output_fn)
    push_forward = DensePushForward(generator)
    dw = random_pvector(model)

    doutput_lin = push_forward.mv(dw)

    output_before = get_output_vector(loader, output_fn)
    update_model(model, dw.get_flat_representation())
    output_after = get_output_vector(loader, output_fn)

    check_tensors(output_after - output_before,
                  doutput_lin.get_flat_representation())


def test_jacobian_pushforward_implicit():
    loader, model, output_fn = get_linear_task()
    generator = Jacobian(model=model,
                         loader=loader,
                         output_fn=output_fn)
    dense_push_forward = DensePushForward(generator)
    implicit_push_forward = DensePushForward(generator)
    dw = random_pvector(model)

    doutput_lin_dense = dense_push_forward.mv(dw)
    doutput_lin_implicit = implicit_push_forward.mv(dw)

    check_tensors(doutput_lin_dense.get_flat_representation(),
                  doutput_lin_implicit.get_flat_representation())
