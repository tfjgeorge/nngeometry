import torch
from tasks import (get_linear_task, get_batchnorm_linear_task,
                   get_fullyconnect_onlylast_task,
                   get_fullyconnect_task, get_batchnorm_nonlinear_task)
from nngeometry.object.map import (PushForwardDense, PushForwardImplicit,
                                   PullBackDense)
from nngeometry.object.fspace import FSpaceDense
from nngeometry.object.pspace import PSpaceDense, PSpaceDiag, PSpaceBlockDiag
from nngeometry.generator import Jacobian
from nngeometry.object.vector import random_pvector, random_fvector, PVector
from utils import check_ratio, check_tensors


linear_tasks = [get_linear_task, get_batchnorm_linear_task,
                get_fullyconnect_onlylast_task]

nonlinear_tasks = [get_fullyconnect_task, get_batchnorm_nonlinear_task]


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
            fspace_dense = FSpaceDense(generator)
            df = random_fvector(len(loader.sampler), n_output, device='cuda')

            # Test get_dense_tensor
            jacobian = pull_back.get_dense_tensor()
            sj = jacobian.size()
            fspace_computed = torch.mm(jacobian.view(-1, sj[2]),
                                       jacobian.view(-1, sj[2]).t())
            check_tensors(fspace_computed.view(sj[0], sj[1], sj[0], sj[1]),
                          fspace_dense.get_dense_tensor(), eps=1e-4)

            # Test vTMv
            vTMv_fspace = fspace_dense.vTMv(df)
            Jv_pullback = pull_back.mv(df).get_flat_representation()
            vTMv_pullforward = torch.dot(Jv_pullback, Jv_pullback)
            check_ratio(vTMv_pullforward, vTMv_fspace)

            # Test frobenius
            frob_fspace = fspace_dense.frobenius_norm()
            frob_direct = (fspace_dense.get_dense_tensor()**2).sum()**.5
            check_ratio(frob_direct, frob_fspace)


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
            pspace_dense = PSpaceDense(generator)
            dw = random_pvector(lc, device='cuda')
            n = len(loader.sampler)

            # Test get_dense_tensor
            jacobian = push_forward.get_dense_tensor()
            sj = jacobian.size()
            pspace_computed = torch.mm(jacobian.view(-1, sj[2]).t(),
                                       jacobian.view(-1, sj[2])) / n
            check_tensors(pspace_computed,
                          pspace_dense.get_dense_tensor(), eps=1e-4)

            # Test get_diag
            check_tensors(torch.diag(pspace_dense.get_dense_tensor()),
                          pspace_dense.get_diag())

            # Test vTMv
            vTMv_pspace = pspace_dense.vTMv(dw)
            Jv_pushforward = push_forward.mv(dw)
            Jv_pushforward_flat = Jv_pushforward.get_flat_representation()
            vTMv_pushforward = torch.dot(Jv_pushforward_flat.view(-1),
                                         Jv_pushforward_flat.view(-1)) / n
            check_ratio(vTMv_pushforward, vTMv_pspace)

            # Test Mv
            Mv_pspace = pspace_dense.mv(dw)
            Mv_pf_pb = pull_back.mv(Jv_pushforward)
            check_tensors(Mv_pf_pb.get_flat_representation() / n,
                          Mv_pspace.get_flat_representation(), eps=1e-4)

            # Test frobenius
            frob_pspace = pspace_dense.frobenius_norm()
            frob_direct = (pspace_dense.get_dense_tensor()**2).sum()**.5
            check_ratio(frob_direct, frob_pspace)

            # Test trace
            trace_pspace = pspace_dense.trace()
            trace_direct = torch.trace(pspace_dense.get_dense_tensor())
            check_ratio(trace_pspace, trace_direct)

            # Test solve
            # NB: regul is very high since the conditioning of pspace_dense
            # is very bad
            regul = 1e0
            Mv_regul = torch.mv(pspace_dense.get_dense_tensor() +
                                regul * torch.eye(pspace_dense.size(0),
                                                  device='cuda'),
                                dw.get_flat_representation())
            Mv_regul = PVector(layer_collection=lc,
                               vector_repr=Mv_regul)
            dw_using_inv = pspace_dense.solve(Mv_regul, regul=1e0)
            check_tensors(dw.get_flat_representation(),
                          dw_using_inv.get_flat_representation(), eps=5e-3)

            # Test inv
            pspace_inv = pspace_dense.inverse(regul=regul)
            check_tensors(dw.get_flat_representation(),
                          pspace_inv.mv(pspace_dense.mv(dw) + regul * dw)
                          .get_flat_representation(), eps=5e-3)


def test_jacobian_pdiag_vs_pdense():
    for get_task in linear_tasks + nonlinear_tasks:
        loader, lc, parameters, model, function, n_output = get_task()
        model.train()
        generator = Jacobian(layer_collection=lc,
                             model=model,
                             loader=loader,
                             function=function,
                             n_output=n_output)
        pspace_diag = PSpaceDiag(generator)
        pspace_dense = PSpaceDense(generator)

        # Test get_dense_tensor
        matrix_diag = pspace_diag.get_dense_tensor()
        matrix_dense = pspace_dense.get_dense_tensor()
        check_tensors(torch.diag(matrix_diag),
                      torch.diag(matrix_dense))
        assert torch.norm(matrix_diag -
                          torch.diag(torch.diag(matrix_diag))) < 1e-5


def test_jacobian_pblockdiag_vs_pdense():
    for get_task in linear_tasks + nonlinear_tasks:
        loader, lc, parameters, model, function, n_output = get_task()
        model.train()
        generator = Jacobian(layer_collection=lc,
                             model=model,
                             loader=loader,
                             function=function,
                             n_output=n_output)
        pspace_blockdiag = PSpaceBlockDiag(generator)
        pspace_dense = PSpaceDense(generator)

        # Test get_dense_tensor
        matrix_blockdiag = pspace_blockdiag.get_dense_tensor()
        matrix_dense = pspace_dense.get_dense_tensor()
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
