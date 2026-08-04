import torch
from tasks import get_conv_gn_task, get_conv_task, get_fullyconnect_task

from nngeometry import Jacobian
from nngeometry.backend import TorchHooksJacobianBackend
from nngeometry.object.fspace import FMatDense
from nngeometry.object.pspace import PMatDense
from nngeometry.object.vector import random_fvector, random_pvector

nonlinear_tasks = [get_conv_gn_task, get_fullyconnect_task, get_conv_task]


def test_jacobian():
    for get_task in nonlinear_tasks:
        loader, lc, parameters, model, function = get_task()
        backend = TorchHooksJacobianBackend(
            model=model,
            function=function,
        )

        FMat_dense = FMatDense(generator=backend, examples=loader, layer_collection=lc)
        PMat_dense = PMatDense(generator=backend, examples=loader, layer_collection=lc)

        jacobian = Jacobian(
            model=model, function=function, loader=loader, layer_collection=lc
        )

        dw = random_pvector(layer_collection=lc)
        df = random_fvector(n_samples=jacobian.size(1), n_output=jacobian.size(0))

        torch.testing.assert_close(
            FMat_dense.to_torch(), (jacobian @ jacobian.adjoint()).to_torch()
        )
        torch.testing.assert_close(
            PMat_dense.to_torch(),
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
