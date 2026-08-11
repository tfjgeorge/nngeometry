import pytest
import torch
from tasks import (
    get_conv_gn_task,
    get_conv_task,
    get_fullyconnect_task,
)
from utils import check_ratio, check_tensors

from nngeometry import GramMatrix, Jacobian
from nngeometry.object.fspace import FMatDense
from nngeometry.object.map import PFMapDense
from nngeometry.object.vector import random_fvector

nonlinear_tasks = [get_conv_gn_task, get_fullyconnect_task, get_conv_task]


def test_gram_vs_jacobian():
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

        torch.testing.assert_close(
            gram.to_torch(), (jacobian @ jacobian.adjoint()).to_torch()
        )
        assert gram.size(0) == gram.size(2)
        assert gram.size(1) == gram.size(3)

        check_ratio(gram.norm(), torch.linalg.norm(gram.to_torch()))

        # __matmul__ API
        torch.testing.assert_close((gram @ gram).to_torch(), (gram**2).to_torch())
        torch.testing.assert_close((gram + gram).to_torch(), (2 * gram).to_torch())
        torch.testing.assert_close((gram - gram + gram).to_torch(), gram.to_torch())
        torch.testing.assert_close(
            (gram.solve(gram, 1e-3)).to_torch(), (gram.inv(1e-3) @ gram).to_torch()
        )

        df = random_fvector(jacobian.size(1), jacobian.size(0))
        torch.testing.assert_close((gram @ df).to_torch(), (df @ gram).to_torch())

        asym_gram = FMatDense(
            lc, gram.generator, data=torch.rand(gram.size(0), gram.size(1), 25, 10)
        )
        torch.testing.assert_close(
            (gram @ asym_gram).adjoint().to_torch(),
            (asym_gram.adjoint() @ gram).to_torch(),
        )

        with pytest.raises(TypeError):
            gram @ 2
        with pytest.raises(TypeError):
            2 @ gram
