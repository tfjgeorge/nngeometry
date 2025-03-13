import torch
from tasks import (
    get_conv_gn_task,
    get_conv_task,
    get_fullyconnect_task,
)

from nngeometry import Jacobian, GramMatrix
from utils import check_tensors

nonlinear_tasks = [get_conv_gn_task, get_fullyconnect_task, get_conv_task]


def test_gram_vs_jacobian():
    for get_task in nonlinear_tasks:
        loader, lc, parameters, model, function = get_task()

        gram = GramMatrix(model=model, function=function, loader=loader)
        jacobian = Jacobian(model=model, function=function, loader=loader)

        jacobian_torch = jacobian.to_torch()
        sj = jacobian_torch.size()
        gram_computed = torch.mm(
            jacobian_torch.view(-1, sj[2]), jacobian_torch.view(-1, sj[2]).t()
        )

        check_tensors(
            gram_computed.view(sj[0], sj[1], sj[0], sj[1]),
            gram.to_torch(),
            eps=1e-4,
        )
