import torch
from tasks import (get_linear_fc_task, get_linear_conv_task,
                   get_batchnorm_fc_linear_task,
                   get_batchnorm_conv_linear_task,
                   get_fullyconnect_onlylast_task,
                   get_fullyconnect_task, get_fullyconnect_bn_task,
                   get_batchnorm_nonlinear_task,
                   get_conv_task, get_conv_bn_task)
from nngeometry.object.map import (PushForwardDense, PushForwardImplicit,
                                   PullBackDense)
from nngeometry.object.fspace import FMatDense
from nngeometry.object.pspace import (PMatDense, PMatDiag, PMatBlockDiag,
                                      PMatImplicit, PMatLowRank)
from nngeometry.generator import Jacobian
from nngeometry.object.vector import random_pvector, random_fvector, PVector
from utils import check_ratio, check_tensors
import pytest


linear_tasks = [get_linear_fc_task, get_linear_conv_task,
                get_batchnorm_fc_linear_task, get_batchnorm_conv_linear_task,
                get_fullyconnect_onlylast_task]

nonlinear_tasks = [get_fullyconnect_task, get_fullyconnect_bn_task,
                   get_conv_bn_task, get_conv_task]


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
    for get_task in [get_fullyconnect_task, get_batchnorm_nonlinear_task]:
        loader, lc, parameters, model, function, n_output = get_task()
        generator = Jacobian(layer_collection=lc,
                             model=model,
                             loader=loader,
                             function=function,
                             n_output=n_output)
        push_forward = PushForwardDense(generator)
        dw = random_pvector(lc, device='cuda')
        dw = 1e-4 / dw.norm() * dw

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


def test_jacobian_fdense_vs_pullback():
    for get_task in linear_tasks + nonlinear_tasks:
        for centering in [True, False]:
            loader, lc, parameters, model, function, n_output = get_task()
            model.train()
            generator = Jacobian(layer_collection=lc,
                                 model=model,
                                 loader=loader,
                                 function=function,
                                 n_output=n_output,
                                 centering=centering)
            pull_back = PullBackDense(generator)
            FMat_dense = FMatDense(generator)
            df = random_fvector(len(loader.sampler), n_output, device='cuda')

            # Test get_dense_tensor
            jacobian = pull_back.get_dense_tensor()
            sj = jacobian.size()
            FMat_computed = torch.mm(jacobian.view(-1, sj[2]),
                                       jacobian.view(-1, sj[2]).t())
            check_tensors(FMat_computed.view(sj[0], sj[1], sj[0], sj[1]),
                          FMat_dense.get_dense_tensor(), eps=1e-4)

            # Test vTMv
            vTMv_FMat = FMat_dense.vTMv(df)
            Jv_pullback = pull_back.mv(df).get_flat_representation()
            vTMv_pullforward = torch.dot(Jv_pullback, Jv_pullback)
            check_ratio(vTMv_pullforward, vTMv_FMat)

            # Test frobenius
            frob_FMat = FMat_dense.frobenius_norm()
            frob_direct = (FMat_dense.get_dense_tensor()**2).sum()**.5
            check_ratio(frob_direct, frob_FMat)


def test_jacobian_pdense_vs_pushforward():
    for get_task in linear_tasks + nonlinear_tasks:
        for centering in [True, False]:
            loader, lc, parameters, model, function, n_output = get_task()
            model.train()
            generator = Jacobian(layer_collection=lc,
                                 model=model,
                                 loader=loader,
                                 function=function,
                                 n_output=n_output,
                                 centering=centering)
            push_forward = PushForwardDense(generator)
            pull_back = PullBackDense(generator, data=push_forward.data)
            PMat_dense = PMatDense(generator)
            dw = random_pvector(lc, device='cuda')
            n = len(loader.sampler)

            # Test get_dense_tensor
            jacobian = push_forward.get_dense_tensor()
            sj = jacobian.size()
            PMat_computed = torch.mm(jacobian.view(-1, sj[2]).t(),
                                       jacobian.view(-1, sj[2])) / n
            check_tensors(PMat_computed,
                          PMat_dense.get_dense_tensor(), eps=1e-4)

            # Test vTMv
            vTMv_PMat = PMat_dense.vTMv(dw)
            Jv_pushforward = push_forward.mv(dw)
            Jv_pushforward_flat = Jv_pushforward.get_flat_representation()
            vTMv_pushforward = torch.dot(Jv_pushforward_flat.view(-1),
                                         Jv_pushforward_flat.view(-1)) / n
            check_ratio(vTMv_pushforward, vTMv_PMat)

            # Test Mv
            Mv_PMat = PMat_dense.mv(dw)
            Mv_pf_pb = pull_back.mv(Jv_pushforward)
            check_tensors(Mv_pf_pb.get_flat_representation() / n,
                          Mv_PMat.get_flat_representation(), eps=1e-4)


def test_jacobian_pdense():
    for get_task in nonlinear_tasks:
        for centering in [True, False]:
            loader, lc, parameters, model, function, n_output = get_task()
            model.train()
            generator = Jacobian(layer_collection=lc,
                                 model=model,
                                 loader=loader,
                                 function=function,
                                 n_output=n_output,
                                 centering=centering)
            PMat_dense = PMatDense(generator)
            dw = random_pvector(lc, device='cuda')

            # Test get_diag
            check_tensors(torch.diag(PMat_dense.get_dense_tensor()),
                          PMat_dense.get_diag())

            # Test frobenius
            frob_PMat = PMat_dense.frobenius_norm()
            frob_direct = (PMat_dense.get_dense_tensor()**2).sum()**.5
            check_ratio(frob_direct, frob_PMat)

            # Test trace
            trace_PMat = PMat_dense.trace()
            trace_direct = torch.trace(PMat_dense.get_dense_tensor())
            check_ratio(trace_PMat, trace_direct)

            # Test solve
            # NB: regul is very high since the conditioning of PMat_dense
            # is very bad
            regul = 1e0
            Mv_regul = torch.mv(PMat_dense.get_dense_tensor() +
                                regul * torch.eye(PMat_dense.size(0),
                                                  device='cuda'),
                                dw.get_flat_representation())
            Mv_regul = PVector(layer_collection=lc,
                               vector_repr=Mv_regul)
            dw_using_inv = PMat_dense.solve(Mv_regul, regul=regul)
            check_tensors(dw.get_flat_representation(),
                          dw_using_inv.get_flat_representation(), eps=5e-3)

            # Test inv
            PMat_inv = PMat_dense.inverse(regul=regul)
            check_tensors(dw.get_flat_representation(),
                          PMat_inv.mv(PMat_dense.mv(dw) + regul * dw)
                          .get_flat_representation(), eps=5e-3)

            # Test add, sub, rmul
            loader, lc, parameters, model, function, n_output = get_task()
            model.train()
            generator = Jacobian(layer_collection=lc,
                                 model=model,
                                 loader=loader,
                                 function=function,
                                 n_output=n_output,
                                 centering=centering)
            PMat_dense2 = PMatDense(generator)

            check_tensors(PMat_dense.get_dense_tensor() +
                          PMat_dense2.get_dense_tensor(),
                          (PMat_dense + PMat_dense2).get_dense_tensor())
            check_tensors(PMat_dense.get_dense_tensor() -
                          PMat_dense2.get_dense_tensor(),
                          (PMat_dense - PMat_dense2).get_dense_tensor())
            check_tensors(1.23 * PMat_dense.get_dense_tensor(),
                          (1.23 * PMat_dense).get_dense_tensor())


def test_jacobian_pdiag_vs_pdense():
    for get_task in nonlinear_tasks:
        loader, lc, parameters, model, function, n_output = get_task()
        model.train()
        generator = Jacobian(layer_collection=lc,
                             model=model,
                             loader=loader,
                             function=function,
                             n_output=n_output)
        PMat_diag = PMatDiag(generator)
        PMat_dense = PMatDense(generator)
        dw = random_pvector(lc, device='cuda')

        # Test get_dense_tensor
        matrix_diag = PMat_diag.get_dense_tensor()
        matrix_dense = PMat_dense.get_dense_tensor()
        check_tensors(torch.diag(matrix_diag),
                      torch.diag(matrix_dense))
        assert torch.norm(matrix_diag -
                          torch.diag(torch.diag(matrix_diag))) < 1e-5

        # Test trace
        check_ratio(torch.trace(matrix_diag),
                    PMat_diag.trace())

        # Test frobenius
        check_ratio(torch.norm(matrix_diag),
                    PMat_diag.frobenius_norm())

        # Test mv
        mv_direct = torch.mv(matrix_diag, dw.get_flat_representation())
        mv_PMat_diag = PMat_diag.mv(dw)
        check_tensors(mv_direct,
                      mv_PMat_diag.get_flat_representation())

        # Test vTMv
        vTMv_direct = torch.dot(mv_direct, dw.get_flat_representation())
        vTMv_PMat_diag = PMat_diag.vTMv(dw)
        check_ratio(vTMv_direct, vTMv_PMat_diag)

        # Test inverse
        regul = 1e-3
        PMat_diag_inverse = PMat_diag.inverse(regul)
        prod = torch.mm(matrix_diag + regul * torch.eye(lc.numel(),
                                                        device='cuda'),
                        PMat_diag_inverse.get_dense_tensor())
        check_tensors(torch.eye(lc.numel(), device='cuda'),
                      prod)

        # Test get_diag
        diag_direct = torch.diag(matrix_diag)
        diag_PMat_diag = PMat_diag.get_diag()
        check_tensors(diag_direct, diag_PMat_diag)

        # Test add, sub, rmul
        loader, lc, parameters, model, function, n_output = get_task()
        model.train()
        generator = Jacobian(layer_collection=lc,
                             model=model,
                             loader=loader,
                             function=function,
                             n_output=n_output)
        PMat_diag2 = PMatDiag(generator)

        check_tensors(PMat_diag.get_dense_tensor() +
                      PMat_diag2.get_dense_tensor(),
                      (PMat_diag + PMat_diag2).get_dense_tensor())
        check_tensors(PMat_diag.get_dense_tensor() -
                      PMat_diag2.get_dense_tensor(),
                      (PMat_diag - PMat_diag2).get_dense_tensor())
        check_tensors(1.23 * PMat_diag.get_dense_tensor(),
                      (1.23 * PMat_diag).get_dense_tensor())


def test_jacobian_pblockdiag_vs_pdense():
    for get_task in linear_tasks + nonlinear_tasks:
        loader, lc, parameters, model, function, n_output = get_task()
        model.train()
        generator = Jacobian(layer_collection=lc,
                             model=model,
                             loader=loader,
                             function=function,
                             n_output=n_output)
        PMat_blockdiag = PMatBlockDiag(generator)
        PMat_dense = PMatDense(generator)

        # Test get_dense_tensor
        matrix_blockdiag = PMat_blockdiag.get_dense_tensor()
        matrix_dense = PMat_dense.get_dense_tensor()
        for layer_id, layer in lc.layers.items():
            start = lc.p_pos[layer_id]
            # compare blocks
            check_tensors(matrix_dense[start:start+layer.numel(),
                                       start:start+layer.numel()],
                          matrix_blockdiag[start:start+layer.numel(),
                                           start:start+layer.numel()])
            # verify that the rest is 0
            assert torch.norm(matrix_blockdiag[start:start+layer.numel(),
                                               start+layer.numel():]) < 1e-5
            assert torch.norm(matrix_blockdiag[start+layer.numel():,
                                               start:start+layer.numel()]) \
                < 1e-5


def test_jacobian_pblockdiag():
    for get_task in nonlinear_tasks:
        loader, lc, parameters, model, function, n_output = get_task()
        model.train()
        generator = Jacobian(layer_collection=lc,
                             model=model,
                             loader=loader,
                             function=function,
                             n_output=n_output)
        PMat_blockdiag = PMatBlockDiag(generator)
        dw = random_pvector(lc, device='cuda')
        dense_tensor = PMat_blockdiag.get_dense_tensor()

        # Test get_diag
        check_tensors(torch.diag(dense_tensor),
                      PMat_blockdiag.get_diag())

        # Test frobenius
        frob_PMat = PMat_blockdiag.frobenius_norm()
        frob_direct = (dense_tensor**2).sum()**.5
        check_ratio(frob_direct, frob_PMat)

        # Test trace
        trace_PMat = PMat_blockdiag.trace()
        trace_direct = torch.trace(dense_tensor)
        check_ratio(trace_PMat, trace_direct)

        # Test mv
        mv_direct = torch.mv(dense_tensor, dw.get_flat_representation())
        check_tensors(mv_direct,
                      PMat_blockdiag.mv(dw).get_flat_representation())

        # Test vTMV
        check_ratio(torch.dot(mv_direct, dw.get_flat_representation()),
                    PMat_blockdiag.vTMv(dw))

        # Test solve
        # NB: regul is very high since the conditioning of PMat_blockdiag
        # is very bad
        regul = 1e0
        # Mv_regul = torch.mv(PMat_blockdiag.get_dense_tensor() +
        #                     regul * torch.eye(PMat_blockdiag.size(0),
        #                                       device='cuda'),
        #                     dw.get_flat_representation())
        # Mv_regul = PVector(layer_collection=lc,
        #                    vector_repr=Mv_regul)
        # dw_using_inv = PMat_blockdiag.solve(Mv_regul, regul=1e0)
        # check_tensors(dw.get_flat_representation(),
        #               dw_using_inv.get_flat_representation(), eps=5e-3)

        # Test inv
        PMat_inv = PMat_blockdiag.inverse(regul=regul)
        check_tensors(dw.get_flat_representation(),
                      PMat_inv.mv(PMat_blockdiag.mv(dw) + regul * dw)
                      .get_flat_representation(), eps=5e-3)

        # Test add, sub, rmul
        loader, lc, parameters, model, function, n_output = get_task()
        model.train()
        generator = Jacobian(layer_collection=lc,
                             model=model,
                             loader=loader,
                             function=function,
                             n_output=n_output)
        PMat_blockdiag2 = PMatBlockDiag(generator)

        check_tensors(PMat_blockdiag.get_dense_tensor() +
                      PMat_blockdiag2.get_dense_tensor(),
                      (PMat_blockdiag + PMat_blockdiag2)
                      .get_dense_tensor())
        check_tensors(PMat_blockdiag.get_dense_tensor() -
                      PMat_blockdiag2.get_dense_tensor(),
                      (PMat_blockdiag - PMat_blockdiag2)
                      .get_dense_tensor())
        check_tensors(1.23 * PMat_blockdiag.get_dense_tensor(),
                      (1.23 * PMat_blockdiag).get_dense_tensor())


def test_jacobian_pimplicit_vs_pdense():
    for get_task in linear_tasks + nonlinear_tasks:
        loader, lc, parameters, model, function, n_output = get_task()
        model.train()
        generator = Jacobian(layer_collection=lc,
                             model=model,
                             loader=loader,
                             function=function,
                             n_output=n_output)
        PMat_implicit = PMatImplicit(generator)
        PMat_dense = PMatDense(generator)
        dw = random_pvector(lc, device='cuda')

        # Test trace
        check_ratio(PMat_dense.trace(),
                    PMat_implicit.trace())

        # Test mv
        if ('BatchNorm1dLayer' in [l.__class__.__name__
                                   for l in lc.layers.values()] or
            'BatchNorm2dLayer' in [l.__class__.__name__
                                   for l in lc.layers.values()]):
            with pytest.raises(NotImplementedError):
                PMat_implicit.mv(dw)
        else:
            check_tensors(PMat_dense.mv(dw).get_flat_representation(),
                          PMat_implicit.mv(dw).get_flat_representation())

        # Test vTMv
        if ('BatchNorm1dLayer' in [l.__class__.__name__
                                   for l in lc.layers.values()] or
            'BatchNorm2dLayer' in [l.__class__.__name__
                                   for l in lc.layers.values()]):
            with pytest.raises(NotImplementedError):
                PMat_implicit.mv(dw)
        else:
            check_ratio(PMat_dense.vTMv(dw),
                        PMat_implicit.vTMv(dw))


def test_jacobian_plowrank_vs_pdense():
    for get_task in nonlinear_tasks:
        loader, lc, parameters, model, function, n_output = get_task()
        model.train()
        generator = Jacobian(layer_collection=lc,
                             model=model,
                             loader=loader,
                             function=function,
                             n_output=n_output)
        PMat_lowrank = PMatLowRank(generator)
        PMat_dense = PMatDense(generator)

        # Test get_dense_tensor
        matrix_lowrank = PMat_lowrank.get_dense_tensor()
        matrix_dense = PMat_dense.get_dense_tensor()
        check_tensors(matrix_dense,
                      matrix_lowrank)


def test_jacobian_plowrank():
    for get_task in nonlinear_tasks:
        loader, lc, parameters, model, function, n_output = get_task()
        model.train()
        generator = Jacobian(layer_collection=lc,
                             model=model,
                             loader=loader,
                             function=function,
                             n_output=n_output)
        PMat_lowrank = PMatLowRank(generator)
        dw = random_pvector(lc, device='cuda')
        dense_tensor = PMat_lowrank.get_dense_tensor()

        # Test get_diag
        check_tensors(torch.diag(dense_tensor),
                      PMat_lowrank.get_diag(),
                      eps=1e-4)

        # Test frobenius
        frob_PMat = PMat_lowrank.frobenius_norm()
        frob_direct = (dense_tensor**2).sum()**.5
        check_ratio(frob_direct, frob_PMat)

        # Test trace
        trace_PMat = PMat_lowrank.trace()
        trace_direct = torch.trace(dense_tensor)
        check_ratio(trace_PMat, trace_direct)

        # Test mv
        mv_direct = torch.mv(dense_tensor, dw.get_flat_representation())
        check_tensors(mv_direct,
                      PMat_lowrank.mv(dw).get_flat_representation())

        # Test vTMV
        check_ratio(torch.dot(mv_direct, dw.get_flat_representation()),
                    PMat_lowrank.vTMv(dw))

        # Test solve TODO
        # Test inv TODO

        # Test add, sub, rmul

        check_tensors(1.23 * PMat_lowrank.get_dense_tensor(),
                      (1.23 * PMat_lowrank).get_dense_tensor())