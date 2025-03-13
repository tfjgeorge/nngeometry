import torch
from nngeometry.object.fspace import FMatDense
from tasks import (
    device,
    get_conv_gn_task,
    get_conv_task,
    get_fullyconnect_task,
)

from nngeometry.backend import TorchHooksJacobianBackend
from nngeometry import Jacobian
from utils import check_tensors

nonlinear_tasks = [get_conv_gn_task, get_fullyconnect_task, get_conv_task]


def test_jacobian_vs_fdense():
    for get_task in nonlinear_tasks:
        loader, lc, parameters, model, function = get_task()
        backend = TorchHooksJacobianBackend(
            layer_collection=lc,
            model=model,
            function=function,
        )

        FMat_dense = FMatDense(generator=backend, examples=loader)

        jacobian = Jacobian(model=model, function=function, loader=loader)
        jacobian_torch = jacobian.to_torch()
        sj = jacobian_torch.size()
        FMat_computed = torch.mm(
            jacobian_torch.view(-1, sj[2]), jacobian_torch.view(-1, sj[2]).t()
        )

        check_tensors(
            FMat_computed.view(sj[0], sj[1], sj[0], sj[1]),
            FMat_dense.to_torch(),
            eps=1e-4,
        )
