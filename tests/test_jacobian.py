import torch
from tasks import get_conv_gn_task, get_conv_task, get_fullyconnect_task
from utils import check_tensors

from nngeometry import Jacobian
from nngeometry.backend import TorchHooksJacobianBackend
from nngeometry.object.fspace import FMatDense

nonlinear_tasks = [get_conv_gn_task, get_fullyconnect_task, get_conv_task]


def test_jacobian_vs_fdense():
    for get_task in nonlinear_tasks:
        loader, lc, parameters, model, function = get_task()
        backend = TorchHooksJacobianBackend(
            model=model,
            function=function,
        )

        FMat_dense = FMatDense(generator=backend, examples=loader, layer_collection=lc)

        jacobian = Jacobian(
            model=model, function=function, loader=loader, layer_collection=lc
        )
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
