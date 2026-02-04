from functools import partial

import pytest
import torch
from tasks import (
    device,
    get_batchnorm_conv_linear_task,
    get_batchnorm_fc_linear_task,
    get_conv1d_task,
    get_conv_gn_task,
    get_conv_skip_task,
    get_conv_task,
    get_embedding_task,
    get_fullyconnect_affine_task,
    get_fullyconnect_cosine_task,
    get_fullyconnect_onlylast_task,
    get_fullyconnect_task,
    get_fullyconnect_wn_task,
    get_layernorm_3d_task,
    get_layernorm_conv_task,
    get_layernorm_task,
    get_linear_3d_task,
    get_linear_conv_task,
    get_linear_fc_task,
    get_small_conv_transpose_task,
    get_small_conv_wn_task,
)
from torch import nn
from utils import check_ratio, check_tensors, get_output_vector, update_model

from nngeometry import Jacobian
from nngeometry.backend import TorchHooksJacobianBackend
from nngeometry.object.fspace import FMatDense
from nngeometry.object.map import PFMapDense, PFMapImplicit
from nngeometry.object.pspace import (
    PMatBlockDiag,
    PMatDense,
    PMatDiag,
    PMatImplicit,
    PMatLowRank,
    PMatQuasiDiag,
)
from nngeometry.object.vector import PVector, random_fvector, random_pvector

linear_tasks = [
    get_layernorm_3d_task,
    partial(get_layernorm_3d_task, rms_norm=True, bias=False),
    get_embedding_task,
    get_linear_fc_task,
    get_linear_conv_task,
    get_batchnorm_fc_linear_task,
    get_batchnorm_conv_linear_task,
    get_fullyconnect_onlylast_task,
]

nonlinear_tasks = [
    get_linear_3d_task,
    get_layernorm_conv_task,
    partial(get_layernorm_conv_task, bias=False),
    partial(get_layernorm_conv_task, rms_norm=True, bias=False),
    get_layernorm_task,
    partial(get_layernorm_task, bias=False),
    partial(get_layernorm_task, rms_norm=True, bias=False),
    get_conv1d_task,
    get_small_conv_transpose_task,
    get_conv_task,
    get_fullyconnect_affine_task,
    get_fullyconnect_cosine_task,
    get_conv_skip_task,
    get_fullyconnect_wn_task,
    get_small_conv_wn_task,
    get_conv_gn_task,
    get_fullyconnect_task,
]


@pytest.fixture(autouse=True)
def make_test_deterministic():
    torch.manual_seed(1234)
    yield


def test_jacobian_pushforward_dense_linear():
    for get_task in linear_tasks:
        loader, lc, parameters, model, function = get_task()
        generator = TorchHooksJacobianBackend(
            model=model,
            function=function,
        )
        push_forward = PFMapDense(
            generator=generator, examples=loader, layer_collection=lc
        )
        dw = random_pvector(layer_collection=lc, device=device)

        doutput_lin = push_forward.jvp(dw)

        output_before = get_output_vector(loader, function)
        update_model(parameters, dw.to_torch())
        output_after = get_output_vector(loader, function)

        check_tensors(output_after - output_before, doutput_lin.to_torch().t())


def test_jacobian_pushforward_dense_nonlinear():
    for get_task in nonlinear_tasks:
        loader, lc, parameters, model, function = get_task()
        generator = TorchHooksJacobianBackend(
            model=model,
            function=function,
        )
        push_forward = PFMapDense(
            generator=generator, examples=loader, layer_collection=lc
        )
        dw = random_pvector(layer_collection=lc, device=device)
        dw = 1e-5 / dw.norm() * dw

        doutput_lin = push_forward.jvp(dw)

        output_before = get_output_vector(loader, function)
        update_model(parameters, dw.to_torch())
        output_after = get_output_vector(loader, function)

        # This is non linear, so we don't expect the finite difference
        # estimate to be very accurate. We use a larger eps value
        check_tensors(
            output_after - output_before,
            doutput_lin.to_torch().t(),
            eps=5e-3,
        )


def test_jacobian_pushforward_implicit():
    for get_task in linear_tasks:
        loader, lc, parameters, model, function = get_task()
        generator = TorchHooksJacobianBackend(
            model=model,
            function=function,
        )
        dense_push_forward = PFMapDense(
            generator=generator, examples=loader, layer_collection=lc
        )
        implicit_push_forward = PFMapImplicit(
            generator=generator, examples=loader, layer_collection=lc
        )
        dw = random_pvector(layer_collection=lc, device=device)

        doutput_lin_dense = dense_push_forward.jvp(dw)
        doutput_lin_implicit = implicit_push_forward.jvp(dw)

        check_tensors(
            doutput_lin_dense.to_torch(),
            doutput_lin_implicit.to_torch(),
        )


def test_jacobian_pullback_dense():
    for get_task in linear_tasks:
        loader, lc, parameters, model, function = get_task()
        generator = TorchHooksJacobianBackend(
            model=model,
            function=function,
        )
        pull_back = PFMapDense(
            generator=generator, examples=loader, layer_collection=lc
        )
        push_forward = PFMapDense(
            generator=generator, examples=loader, layer_collection=lc
        )
        dw = random_pvector(layer_collection=lc, device=device)

        doutput_lin = push_forward.jvp(dw)
        dinput_lin = pull_back.vjp(doutput_lin)
        check_ratio(
            torch.dot(dw.to_torch(), dinput_lin.to_torch()),
            torch.norm(doutput_lin.to_torch()) ** 2,
        )


def test_jacobian_fdense_vs_pullback():
    for get_task in linear_tasks + nonlinear_tasks:
        for centering in [True, False]:
            loader, lc, parameters, model, function = get_task()
            generator = TorchHooksJacobianBackend(
                model=model,
                function=function,
                centering=centering,
            )
            pull_back = PFMapDense(
                generator=generator, examples=loader, layer_collection=lc
            )
            FMat_dense = FMatDense(
                generator=generator, examples=loader, layer_collection=lc
            )

            n_output = FMat_dense.to_torch().size(0)
            df = random_fvector(len(loader.sampler), n_output, device=device)

            # Test to_torch to get dense tensor
            jacobian = pull_back.to_torch()
            sj = jacobian.size()
            FMat_computed = torch.mm(
                jacobian.view(-1, sj[2]), jacobian.view(-1, sj[2]).t()
            )
            check_tensors(
                FMat_computed.view(sj[0], sj[1], sj[0], sj[1]),
                FMat_dense.to_torch(),
                eps=1e-4,
            )

            # Test vTMv
            vTMv_FMat = FMat_dense.vTMv(df)
            Jv_pullback = pull_back.vjp(df).to_torch()
            vTMv_pullforward = torch.dot(Jv_pullback, Jv_pullback)
            check_ratio(vTMv_pullforward, vTMv_FMat)

            # Test frobenius
            frob_FMat = FMat_dense.norm()
            frob_direct = (FMat_dense.to_torch() ** 2).sum() ** 0.5
            check_ratio(frob_direct, frob_FMat)

            with pytest.raises(RuntimeError):
                FMat_dense.norm("prout")


def test_jacobian_eigendecomposition_fdense():
    for get_task in [get_small_conv_transpose_task]:
        for impl in ["eigh", "svd"]:
            loader, lc, parameters, model, function = get_task()
            generator = TorchHooksJacobianBackend(
                model=model,
                function=function,
                centering=True,
            )
            FMat_dense = FMatDense(
                generator=generator, examples=loader, layer_collection=lc
            )
            FMat_dense.compute_eigendecomposition(impl=impl)
            evals, evecs = FMat_dense.get_eigendecomposition()

            tensor = FMat_dense.to_torch()
            s = tensor.size()
            check_tensors(
                tensor.view(s[0] * s[1], s[2] * s[3]),
                evecs @ torch.diag_embed(evals) @ evecs.T,
            )

        with pytest.raises(NotImplementedError):
            FMat_dense.compute_eigendecomposition(impl="stupid")


def test_jacobian_eigendecomposition_pdense():
    for get_task in [get_small_conv_transpose_task]:
        for impl in ["eigh", "svd"]:
            loader, lc, parameters, model, function = get_task()
            generator = TorchHooksJacobianBackend(
                model=model,
                function=function,
                centering=True,
            )
            pmat_dense = PMatDense(
                generator=generator, examples=loader, layer_collection=lc
            )
            pmat_dense.compute_eigendecomposition(impl=impl)
            evals, evecs = pmat_dense.get_eigendecomposition()

            check_tensors(
                pmat_dense.to_torch(), evecs @ torch.diag_embed(evals) @ evecs.T
            )

        with pytest.raises(NotImplementedError):
            pmat_dense.compute_eigendecomposition(impl="stupid")


def test_jacobian_eigendecomposition_plowrank():
    for get_task in [get_conv_task]:
        for impl in ["svd"]:
            loader, lc, parameters, model, function = get_task()
            generator = TorchHooksJacobianBackend(
                model=model,
                function=function,
                centering=True,
            )
            pmat_lowrank = PMatLowRank(
                generator=generator, examples=loader, layer_collection=lc
            )
            pmat_lowrank.compute_eigendecomposition(impl=impl)
            evals, evecs = pmat_lowrank.get_eigendecomposition()

            assert not evals.isnan().any()
            assert not evecs.isnan().any()

            check_tensors(
                pmat_lowrank.to_torch(),
                evecs @ torch.diag_embed(evals) @ evecs.T,
            )

        with pytest.raises(NotImplementedError):
            pmat_lowrank.compute_eigendecomposition(impl="stupid")


def test_jacobian_pdense_vs_pushforward():
    # NB: sometimes the test with centering=True do not pass,
    # which is probably due to the way we compute centering
    # for PMatDense: E[x^2] - (Ex)^2 is notoriously not numerically stable
    for get_task in linear_tasks + nonlinear_tasks:
        for centering in [True, False]:
            loader, lc, parameters, model, function = get_task()
            generator = TorchHooksJacobianBackend(
                model=model,
                function=function,
                centering=centering,
            )
            push_forward = PFMapDense(
                generator=generator, examples=loader, layer_collection=lc
            )
            pull_back = PFMapDense(
                generator=generator, data=push_forward.data, layer_collection=lc
            )
            PMat_dense = PMatDense(
                generator=generator, examples=loader, layer_collection=lc
            )
            dw = random_pvector(layer_collection=lc, device=device)
            n = len(loader.sampler)

            # Test to_torch
            jacobian = push_forward.to_torch()
            sj = jacobian.size()
            PMat_computed = (
                torch.mm(jacobian.view(-1, sj[2]).t(), jacobian.view(-1, sj[2])) / n
            )
            check_tensors(PMat_computed, PMat_dense.to_torch())

            # Test vTMv
            vTMv_PMat = PMat_dense.vTMv(dw)
            Jv_pushforward = push_forward.jvp(dw)
            Jv_pushforward_flat = Jv_pushforward.to_torch()
            vTMv_pushforward = (
                torch.dot(Jv_pushforward_flat.view(-1), Jv_pushforward_flat.view(-1))
                / n
            )
            check_ratio(vTMv_pushforward, vTMv_PMat)

            # Test Mv
            Mv_PMat = PMat_dense.mv(dw)
            Mv_pf_pb = pull_back.vjp(Jv_pushforward)
            check_tensors(
                Mv_pf_pb.to_torch() / n,
                Mv_PMat.to_torch(),
            )


def test_jacobian_pdense():
    for get_task in nonlinear_tasks:
        for centering in [True, False]:
            loader, lc, parameters, model, function = get_task()
            generator = TorchHooksJacobianBackend(
                model=model,
                function=function,
                centering=centering,
            )
            PMat_dense = PMatDense(
                generator=generator, examples=loader, layer_collection=lc
            )
            dw = random_pvector(layer_collection=lc, device=device)

            # Test get_diag
            check_tensors(torch.diag(PMat_dense.to_torch()), PMat_dense.get_diag())

            # Test frobenius
            frob_PMat = PMat_dense.norm()
            frob_direct = (PMat_dense.to_torch() ** 2).sum() ** 0.5
            check_ratio(frob_direct, frob_PMat)

            # Test spectral
            spec_PMat = PMat_dense.norm(2)
            spec_direct = torch.linalg.eigvalsh(PMat_dense.to_torch()).max()
            check_ratio(spec_direct, spec_PMat)

            spec_PMat = PMat_dense.norm(-2)
            spec_direct = torch.linalg.eigvalsh(PMat_dense.to_torch()).min()
            torch.testing.assert_close(spec_PMat, spec_direct)

            with pytest.raises(RuntimeError):
                PMat_dense.norm(ord="prout")

            # Test trace
            trace_PMat = PMat_dense.trace()
            trace_direct = torch.trace(PMat_dense.to_torch())
            check_ratio(trace_PMat, trace_direct)

            # Test solve
            # NB: regul is high since the matrix is not full rank
            regul = 1e-3
            Mv_torch = torch.mv(PMat_dense.to_torch(), dw.to_torch())
            Mv_regul = PVector(
                layer_collection=lc, vector_repr=Mv_torch + regul * dw.to_torch()
            )
            dw_solve = PMat_dense.solve(Mv_regul, regul=regul)
            check_tensors(
                dw.to_torch(),
                dw_solve.to_torch(),
                eps=5e-3,
            )

            # Test solve with jacobian
            # TODO improve
            c = 1.678
            stacked_mv = torch.stack([c**i * Mv_torch for i in range(6)]).reshape(
                2, 3, -1
            )
            stacked_v = torch.stack([c**i * dw.to_torch() for i in range(6)]).reshape(
                2, 3, -1
            )
            jaco = PFMapDense(
                generator=generator,
                data=stacked_mv + regul * stacked_v,
                layer_collection=lc,
            )
            J_back = PMat_dense.solve(jaco, regul=regul)

            check_tensors(
                stacked_v,
                J_back.to_torch(),
            )

            # Test inv
            PMat_inv = PMat_dense.inv(regul=regul)
            check_tensors(
                dw.to_torch(),
                PMat_inv.mv(PMat_dense.mv(dw) + regul * dw).to_torch(),
                eps=5e-3,
            )

            # Test add, sub, rmul
            loader, lc, parameters, model, function = get_task()
            generator = TorchHooksJacobianBackend(
                model=model,
                function=function,
                centering=centering,
            )
            PMat_dense2 = PMatDense(
                generator=generator, examples=loader, layer_collection=lc
            )

            check_tensors(
                PMat_dense.to_torch() + PMat_dense2.to_torch(),
                (PMat_dense + PMat_dense2).to_torch(),
            )
            check_tensors(
                PMat_dense.to_torch() - PMat_dense2.to_torch(),
                (PMat_dense - PMat_dense2).to_torch(),
            )
            check_tensors(
                1.23 * PMat_dense.to_torch(),
                (1.23 * PMat_dense).to_torch(),
            )


def test_jacobian_pdiag_vs_pdense():
    for get_task in nonlinear_tasks:
        loader, lc, parameters, model, function = get_task()
        generator = TorchHooksJacobianBackend(
            model=model,
            function=function,
        )
        PMat_diag = PMatDiag(generator=generator, examples=loader, layer_collection=lc)
        PMat_dense = PMatDense(
            generator=generator, examples=loader, layer_collection=lc
        )
        dw = random_pvector(layer_collection=lc, device=device)

        # Test to_torch
        matrix_diag = PMat_diag.to_torch()
        matrix_dense = PMat_dense.to_torch()
        check_tensors(torch.diag(matrix_diag), torch.diag(matrix_dense))
        assert torch.norm(matrix_diag - torch.diag(torch.diag(matrix_diag))) < 1e-5

        # Test trace
        check_ratio(torch.trace(matrix_diag), PMat_diag.trace())

        # Test norm
        for ord in ["fro", 2, -2]:
            norm_direct = torch.linalg.norm(matrix_diag, ord=ord)
            norm_diag = PMat_diag.norm(ord=ord)
            torch.testing.assert_close(norm_diag, norm_direct)

        with pytest.raises(RuntimeError):
            PMat_diag.norm(ord="prout")

        # Test mv
        mv_direct = torch.mv(matrix_diag, dw.to_torch())
        mv_PMat_diag = PMat_diag.mv(dw)
        check_tensors(mv_direct, mv_PMat_diag.to_torch())

        # Test vTMv
        vTMv_direct = torch.dot(mv_direct, dw.to_torch())
        vTMv_PMat_diag = PMat_diag.vTMv(dw)
        check_ratio(vTMv_direct, vTMv_PMat_diag)

        # Test inverse
        regul = 1e-3
        PMat_diag_inverse = PMat_diag.inv(regul)
        prod = torch.mm(
            matrix_diag + regul * torch.eye(lc.numel(), device=device),
            PMat_diag_inverse.to_torch(),
        )
        check_tensors(torch.eye(lc.numel(), device=device), prod)

        # Test solve
        regul = 1e-3
        Mv_regul = torch.mv(
            matrix_diag + regul * torch.eye(PMat_diag.size(0), device=device),
            dw.to_torch(),
        )
        Mv_regul = PVector(layer_collection=lc, vector_repr=Mv_regul)
        dw_using_inv = PMat_diag.solve(Mv_regul, regul=regul)
        check_tensors(
            dw.to_torch(),
            dw_using_inv.to_torch(),
            eps=5e-3,
        )

        # Test get_diag
        diag_direct = torch.diag(matrix_diag)
        diag_PMat_diag = PMat_diag.get_diag()
        check_tensors(diag_direct, diag_PMat_diag)

        # Test add, sub, rmul
        loader, lc, parameters, model, function = get_task()
        generator = TorchHooksJacobianBackend(
            model=model,
            function=function,
        )
        PMat_diag2 = PMatDiag(generator=generator, examples=loader, layer_collection=lc)

        check_tensors(
            PMat_diag.to_torch() + PMat_diag2.to_torch(),
            (PMat_diag + PMat_diag2).to_torch(),
        )
        check_tensors(
            PMat_diag.to_torch() - PMat_diag2.to_torch(),
            (PMat_diag - PMat_diag2).to_torch(),
        )
        check_tensors(1.23 * PMat_diag.to_torch(), (1.23 * PMat_diag).to_torch())


def test_jacobian_pblockdiag_vs_pdense():
    for get_task in linear_tasks + nonlinear_tasks:
        loader, lc, parameters, model, function = get_task()
        generator = TorchHooksJacobianBackend(
            model=model,
            function=function,
        )
        PMat_blockdiag = PMatBlockDiag(
            generator=generator, examples=loader, layer_collection=lc
        )
        PMat_dense = PMatDense(
            generator=generator, examples=loader, layer_collection=lc
        )

        # Test to_torch
        matrix_blockdiag = PMat_blockdiag.to_torch()
        matrix_dense = PMat_dense.to_torch()
        for layer_id, layer in lc.layers.items():
            start = lc.p_pos[layer_id]
            # compare blocks
            check_tensors(
                matrix_dense[
                    start : start + layer.numel(), start : start + layer.numel()
                ],
                matrix_blockdiag[
                    start : start + layer.numel(), start : start + layer.numel()
                ],
            )
            # verify that the rest is 0
            assert (
                torch.norm(
                    matrix_blockdiag[
                        start : start + layer.numel(), start + layer.numel() :
                    ]
                )
                < 1e-5
            )
            assert (
                torch.norm(
                    matrix_blockdiag[
                        start + layer.numel() :, start : start + layer.numel()
                    ]
                )
                < 1e-5
            )


def test_jacobian_pblockdiag():
    for get_task in nonlinear_tasks:
        loader, lc, parameters, model, function = get_task()
        generator = TorchHooksJacobianBackend(
            model=model,
            function=function,
        )
        PMat_blockdiag = PMatBlockDiag(
            generator=generator, examples=loader, layer_collection=lc
        )
        dw = random_pvector(layer_collection=lc, device=device)
        dense_tensor = PMat_blockdiag.to_torch()

        # Test get_diag
        check_tensors(torch.diag(dense_tensor), PMat_blockdiag.get_diag())

        # Test norm
        for ord in [-2, 2, "fro"]:
            norm_PMat = PMat_blockdiag.norm(ord=ord)
            norm_direct = torch.linalg.norm(dense_tensor, ord=ord)
            torch.testing.assert_close(norm_direct, norm_PMat)

        with pytest.raises(RuntimeError):
            PMat_blockdiag.norm(ord="prout")

        # Test trace
        trace_PMat = PMat_blockdiag.trace()
        trace_direct = torch.trace(dense_tensor)
        check_ratio(trace_PMat, trace_direct)

        # Test mv
        mv_torch = torch.mv(dense_tensor, dw.to_torch())
        mv_nng = PMat_blockdiag.mv(dw)
        check_tensors(mv_torch, mv_nng.to_torch())

        # Test vTMV
        check_ratio(torch.dot(mv_torch, dw.to_torch()), PMat_blockdiag.vTMv(dw))

        # Test solve
        regul = 1e-3
        Mv_regul = mv_nng + regul * dw
        Mv_regul = PVector(layer_collection=lc, vector_repr=Mv_regul.to_torch())
        dw_using_inv = PMat_blockdiag.solve(Mv_regul, regul=regul)
        check_tensors(
            dw.to_torch(),
            dw_using_inv.to_torch(),
            eps=5e-3,
        )

        # Test solve with jacobian
        # TODO improve
        c = 1.678
        Mv_torch = mv_nng.to_torch()
        stacked_mv = torch.stack([c**i * Mv_torch for i in range(6)]).reshape(2, 3, -1)
        stacked_v = torch.stack([c**i * dw.to_torch() for i in range(6)]).reshape(
            2, 3, -1
        )
        jaco = PFMapDense(
            generator=generator,
            data=stacked_mv + regul * stacked_v,
            layer_collection=lc,
        )
        J_back = PMat_blockdiag.solve(jaco, regul=regul)
        check_tensors(
            stacked_v,
            J_back.to_torch(),
        )

        # Test inv
        PMat_inv = PMat_blockdiag.inv(regul=regul)
        check_tensors(
            dw.to_torch(),
            PMat_inv.mv(PMat_blockdiag.mv(dw) + regul * dw).to_torch(),
            eps=5e-3,
        )

        # Test add, sub, rmul
        loader, lc, parameters, model, function = get_task()
        generator = TorchHooksJacobianBackend(
            model=model,
            function=function,
        )
        PMat_blockdiag2 = PMatBlockDiag(
            generator=generator, examples=loader, layer_collection=lc
        )

        check_tensors(
            PMat_blockdiag.to_torch() + PMat_blockdiag2.to_torch(),
            (PMat_blockdiag + PMat_blockdiag2).to_torch(),
        )
        check_tensors(
            PMat_blockdiag.to_torch() - PMat_blockdiag2.to_torch(),
            (PMat_blockdiag - PMat_blockdiag2).to_torch(),
        )
        check_tensors(
            1.23 * PMat_blockdiag.to_torch(),
            (1.23 * PMat_blockdiag).to_torch(),
        )


def test_jacobian_pimplicit_vs_pdense():
    solve_tested = False

    for get_task in linear_tasks + nonlinear_tasks:
        loader, lc, parameters, model, function = get_task()
        generator = TorchHooksJacobianBackend(
            model=model,
            function=function,
        )
        PMat_implicit = PMatImplicit(
            generator=generator, examples=loader, layer_collection=lc
        )
        PMat_dense = PMatDense(
            generator=generator, examples=loader, layer_collection=lc
        )
        PMat_bd = PMatBlockDiag(
            generator=generator, examples=loader, layer_collection=lc
        )
        dw = random_pvector(layer_collection=lc, device=device)

        # Test trace
        torch.testing.assert_close(PMat_dense.trace(), PMat_implicit.trace())

        # Test mv
        if "BatchNorm1dLayer" in [
            l.__class__.__name__ for l in lc.layers.values()
        ] or "BatchNorm2dLayer" in [l.__class__.__name__ for l in lc.layers.values()]:
            with pytest.raises(NotImplementedError):
                PMat_implicit.mv(dw)
        else:
            torch.testing.assert_close(
                PMat_dense.mv(dw).to_torch(),
                PMat_implicit.mv(dw).to_torch(),
            )

        # Test vTMv
        if "BatchNorm1dLayer" in [
            l.__class__.__name__ for l in lc.layers.values()
        ] or "BatchNorm2dLayer" in [l.__class__.__name__ for l in lc.layers.values()]:
            with pytest.raises(NotImplementedError):
                PMat_implicit.vTMv(dw)
        else:
            torch.testing.assert_close(PMat_dense.vTMv(dw), PMat_implicit.vTMv(dw))

        # Test solvePVec
        regul = 1e-3
        if "BatchNorm1dLayer" in [
            l.__class__.__name__ for l in lc.layers.values()
        ] or "BatchNorm2dLayer" in [l.__class__.__name__ for l in lc.layers.values()]:
            with pytest.raises(NotImplementedError):
                PMat_implicit.solvePVec(dw)
        elif lc.numel() < 100:  # this is a slow test, it is applied only to small nets
            torch.testing.assert_close(
                PMat_dense.solve(dw, regul=regul).to_torch(),
                PMat_implicit.solve(
                    dw, regul=regul, M=PMat_dense, max_iter=1
                ).to_torch(),
                atol=1e-3,
                rtol=1e-3,
            )  # perfect preconditioner
            torch.testing.assert_close(
                PMat_dense.solve(dw, regul=regul).to_torch(),
                PMat_implicit.solve(
                    dw, regul=regul, x0=PMat_dense.solve(dw, regul=regul), max_iter=1
                ).to_torch(),
                atol=1e-3,
                rtol=1e-3,
            )  # good init
            torch.testing.assert_close(
                PMat_dense.solve(dw, regul=regul).to_torch(),
                PMat_implicit.solve(
                    dw, regul=regul, max_iter=100, M=PMat_bd, x0=dw, rtol=1e-3
                ).to_torch(),
                atol=1e-3,
                rtol=1e-3,
            )  # worse preconditioner
            solve_tested = True

    assert solve_tested


def test_jacobian_plowrank_vs_pdense():
    for get_task in nonlinear_tasks:
        loader, lc, parameters, model, function = get_task()
        generator = TorchHooksJacobianBackend(
            model=model,
            function=function,
        )
        PMat_lowrank = PMatLowRank(
            generator=generator, examples=loader, layer_collection=lc
        )
        PMat_dense = PMatDense(
            generator=generator, examples=loader, layer_collection=lc
        )

        # Test to_torch
        matrix_lowrank = PMat_lowrank.to_torch()
        matrix_dense = PMat_dense.to_torch()
        check_tensors(matrix_dense, matrix_lowrank)


def test_jacobian_plowrank():
    for get_task in nonlinear_tasks:
        loader, lc, parameters, model, function = get_task()
        generator = TorchHooksJacobianBackend(
            model=model,
            function=function,
        )
        PMat_lowrank = PMatLowRank(
            generator=generator, examples=loader, layer_collection=lc
        )
        dw = random_pvector(layer_collection=lc, device=device)
        dw = dw / dw.norm()
        dense_tensor = PMat_lowrank.to_torch()

        # Test get_diag
        check_tensors(torch.diag(dense_tensor), PMat_lowrank.get_diag(), eps=1e-4)

        # Test norm
        for ord in [2, "fro"]:
            norm_PMat = PMat_lowrank.norm(ord=ord)
            norm_direct = torch.linalg.norm(dense_tensor, ord=ord)
            check_ratio(norm_direct, norm_PMat)

        with pytest.raises(RuntimeError):
            PMat_lowrank.norm(ord="prout")

        # Test trace
        trace_PMat = PMat_lowrank.trace()
        trace_direct = torch.trace(dense_tensor)
        check_ratio(trace_PMat, trace_direct)

        # Test mv
        mv_direct = torch.mv(dense_tensor, dw.to_torch())
        mv = PMat_lowrank.mv(dw)
        check_tensors(mv_direct, mv.to_torch())

        # Test vTMV
        check_ratio(torch.dot(mv_direct, dw.to_torch()), PMat_lowrank.vTMv(dw))

        # Test solve
        # We will try to recover mv, which is in the span of the
        # low rank matrix
        regul = 1e-3
        mmv = PMat_lowrank.mv(mv)
        mv_using_inv = PMat_lowrank.solve(mmv + regul * mv, regul=regul)
        check_tensors(
            mv.to_torch(),
            mv_using_inv.to_torch(),
            eps=1e-2,
        )

        # Test inv TODO

        # Test add, sub, rmul

        check_tensors(
            1.23 * PMat_lowrank.to_torch(),
            (1.23 * PMat_lowrank).to_torch(),
        )


def test_jacobian_pquasidiag_vs_pdense():
    for get_task in [get_conv_task, get_fullyconnect_task]:
        loader, lc, parameters, model, function = get_task()
        generator = TorchHooksJacobianBackend(
            model=model,
            function=function,
        )
        PMat_qd = PMatQuasiDiag(
            generator=generator, examples=loader, layer_collection=lc
        )
        PMat_dense = PMatDense(
            generator=generator, examples=loader, layer_collection=lc
        )

        # Test to_torch
        matrix_qd = PMat_qd.to_torch()
        matrix_dense = PMat_dense.to_torch()

        for layer_id, layer in lc.layers.items():
            start = lc.p_pos[layer_id]
            # compare diags
            sw = layer.weight.numel()

            check_tensors(
                torch.diag(
                    torch.diag(matrix_dense[start : start + sw, start : start + sw])
                ),
                matrix_qd[start : start + sw, start : start + sw],
            )

            if layer.has_bias():
                sb = layer.bias.numel()
                check_tensors(
                    torch.diag(
                        torch.diag(
                            matrix_dense[
                                start + sw : start + sw + sb,
                                start + sw : start + sw + sb,
                            ]
                        )
                    ),
                    matrix_qd[
                        start + sw : start + sw + sb, start + sw : start + sw + sb
                    ],
                )

                s_in = sw // sb
                for i in range(sb):
                    # check the strips bias/weight
                    check_tensors(
                        matrix_dense[
                            start + i * s_in : start + (i + 1) * s_in, start + sw + i
                        ],
                        matrix_qd[
                            start + i * s_in : start + (i + 1) * s_in, start + sw + i
                        ],
                    )

                    # verify that the rest is 0
                    assert (
                        torch.norm(
                            matrix_qd[
                                start + i * s_in : start + (i + 1) * s_in,
                                start + sw : start + sw + i,
                            ]
                        )
                        < 1e-10
                    )
                    assert (
                        torch.norm(
                            matrix_qd[
                                start + i * s_in : start + (i + 1) * s_in,
                                start + sw + i + 1 :,
                            ]
                        )
                        < 1e-10
                    )

                # compare upper triangular block with lower triangular one
                check_tensors(
                    matrix_qd[start : start + sw + sb, start + sw :],
                    matrix_qd[start + sw :, start : start + sw + sb].t(),
                )


def test_jacobian_pquasidiag():
    for get_task in [get_conv_task, get_fullyconnect_task]:
        loader, lc, parameters, model, function = get_task()
        generator = TorchHooksJacobianBackend(
            model=model,
            function=function,
        )
        PMat_qd = PMatQuasiDiag(
            generator=generator, examples=loader, layer_collection=lc
        )
        dense_tensor = PMat_qd.to_torch()

        v = random_pvector(layer_collection=lc, device=device)
        v_flat = v.to_torch()

        check_tensors(torch.diag(dense_tensor), PMat_qd.get_diag())

        check_ratio(torch.linalg.norm(dense_tensor), PMat_qd.norm())
        with pytest.raises(RuntimeError):
            PMat_qd.norm("prout")

        check_ratio(torch.trace(dense_tensor), PMat_qd.trace())

        mv = PMat_qd.mv(v)
        check_tensors(torch.mv(dense_tensor, v_flat), mv.to_torch())

        check_ratio(torch.dot(torch.mv(dense_tensor, v_flat), v_flat), PMat_qd.vTMv(v))

        # Test solve
        regul = 1e-8
        v_back = PMat_qd.solve(mv + regul * v, regul=regul)
        check_tensors(v.to_torch(), v_back.to_torch())


def test_bn_eval_mode():
    for get_task in [get_batchnorm_fc_linear_task, get_batchnorm_conv_linear_task]:
        loader, lc, parameters, model, function = get_task()

        generator = TorchHooksJacobianBackend(
            model=model,
            function=function,
        )

        model.eval()
        FMat_dense = FMatDense(
            generator=generator, examples=loader, layer_collection=lc
        )

        model.train()
        with pytest.raises(RuntimeError):
            FMat_dense = FMatDense(
                generator=generator, examples=loader, layer_collection=lc
            )


def test_example_passing():
    # test when passing a minibatch of examples instead of the full dataloader
    for get_task in [get_fullyconnect_task]:
        loader, lc, parameters, model, function = get_task()
        generator = TorchHooksJacobianBackend(
            model=model,
            function=function,
        )

        sum_mats = None
        tot_examples = 0
        for d in iter(loader):
            this_mat = PMatDense(generator=generator, examples=d, layer_collection=lc)
            n_examples = len(d[0])

            if sum_mats is None:
                sum_mats = n_examples * this_mat
            else:
                sum_mats = n_examples * this_mat + sum_mats

            tot_examples += n_examples

        PMat_dense = PMatDense(
            generator=generator, examples=loader, layer_collection=lc
        )

        check_tensors(
            PMat_dense.to_torch(),
            (1.0 / tot_examples * sum_mats).to_torch(),
        )


def test_shared_parameters():
    # test that shared parameters (e.g. using twice the same layer)
    # is detected and raises an error
    model = nn.Sequential(*[nn.Linear(4, 4)] * 2)
    d = torch.rand((5, 4))

    with pytest.raises(NotImplementedError):
        J = Jacobian(model, (d,), lambda x: model(x))
