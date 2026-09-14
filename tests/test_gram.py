import pytest
import torch
from tasks import (
    get_conv_gn_task,
    get_conv_task,
    get_fullyconnect_task,
)
from utils import check_ratio

from nngeometry import GramMatrix, Jacobian
from nngeometry.object.fspace import FMatDense
from nngeometry.object.map import PFMapDense, random_pfmap
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

        # __op__
        torch.testing.assert_close((gram + gram).to_torch(), (2 * gram).to_torch())
        torch.testing.assert_close((gram - gram + gram).to_torch(), gram.to_torch())
        with pytest.raises(TypeError):
            gram @ 2
        with pytest.raises(TypeError):
            2 @ gram

        # solve
        df1 = random_fvector(jacobian.size(1), jacobian.size(0))
        J = random_pfmap(lc, (jacobian.size(1), jacobian.size(0)))
        torch.testing.assert_close(
            (gram.solve(df1, 1e-3)).to_torch(), (gram.inv(1e-3) @ df1).to_torch()
        )
        torch.testing.assert_close(
            (gram.solve(J, 1e-3)).to_torch(), (gram.inv(1e-3) @ J).to_torch()
        )
        with pytest.raises(NotImplementedError):
            gram.solve(gram, solve="prout")
        with pytest.raises(NotImplementedError):
            gram.solve(df1, solve="prout")

