import torch
import torch.nn.functional as tF
from tasks import (get_linear_fc_task, get_linear_conv_task,
                   get_batchnorm_fc_linear_task,
                   get_batchnorm_conv_linear_task,
                   get_fullyconnect_onlylast_task,
                   get_fullyconnect_task, get_fullyconnect_bn_task,
                   get_batchnorm_nonlinear_task,
                   get_conv_task, get_conv_bn_task, get_conv_gn_task)
from tasks import to_device
from nngeometry.object.map import (PushForwardDense, PushForwardImplicit,
                                   PullBackDense)
from nngeometry.object.fspace import FMatDense
from nngeometry.object.pspace import (PMatDense, PMatDiag, PMatBlockDiag,
                                      PMatImplicit, PMatLowRank, PMatQuasiDiag)
from nngeometry.generator import Jacobian
from nngeometry.metrics import FIM, FIM_MonteCarlo
from nngeometry.object.vector import random_pvector, random_fvector, PVector
from utils import check_ratio, check_tensors, check_angle
from test_jacobian import update_model, get_output_vector, device

nonlinear_tasks = [get_conv_gn_task, get_fullyconnect_task, get_conv_task]


import numpy as np


def test_FIM_MC_vs_linearization():
    step = 1e-2

    for get_task in nonlinear_tasks:
        quots = []
        for i in range(10): # repeat to kill statistical fluctuations
            loader, lc, parameters, model, function, n_output = get_task()
            model.train()
            F = FIM_MonteCarlo(layer_collection=lc,
                               model=model,
                               loader=loader,
                               variant='classif_logits',
                               representation=PMatDense,
                               trials=10,
                               function=lambda *d: model(to_device(d[0])))

            dw = random_pvector(lc, device=device)
            dw = step / dw.norm() * dw

            output_before = get_output_vector(loader, function)
            update_model(parameters, dw.get_flat_representation())
            output_after = get_output_vector(loader, function)
            update_model(parameters, -dw.get_flat_representation())

            KL = tF.kl_div(tF.log_softmax(output_before, dim=1),
                           tF.log_softmax(output_after, dim=1),
                           log_target=True, reduction='batchmean')

            quot = (KL / F.vTMv(dw) * 2) ** .5

            quots.append(quot.item())

        mean_quotient = sum(quots) / len(quots)
        assert mean_quotient > 1 - 5e-2 and mean_quotient < 1 + 5e-2


def test_FIM_vs_linearization_classif_logits():
    step = 1e-2

    for get_task in nonlinear_tasks:
        quots = []
        for i in range(10): # repeat to kill statistical fluctuations
            loader, lc, parameters, model, function, n_output = get_task()
            model.train()
            F = FIM(layer_collection=lc,
                    model=model,
                    loader=loader,
                    variant='classif_logits',
                    representation=PMatDense,
                    n_output=n_output,
                    function=lambda *d: model(to_device(d[0])))

            dw = random_pvector(lc, device=device)
            dw = step / dw.norm() * dw

            output_before = get_output_vector(loader, function)
            update_model(parameters, dw.get_flat_representation())
            output_after = get_output_vector(loader, function)
            update_model(parameters, -dw.get_flat_representation())

            KL = tF.kl_div(tF.log_softmax(output_before, dim=1),
                           tF.log_softmax(output_after, dim=1),
                           log_target=True, reduction='batchmean')

            quot = (KL / F.vTMv(dw) * 2) ** .5

            quots.append(quot.item())

        mean_quotient = sum(quots) / len(quots)
        assert mean_quotient > 1 - 5e-2 and mean_quotient < 1 + 5e-2

def test_FIM_vs_linearization_regression():
    step = 1e-2

    for get_task in nonlinear_tasks:
        quots = []
        for i in range(10): # repeat to kill statistical fluctuations
            loader, lc, parameters, model, function, n_output = get_task()
            model.train()
            F = FIM(layer_collection=lc,
                    model=model,
                    loader=loader,
                    variant='regression',
                    representation=PMatDense,
                    n_output=n_output,
                    function=lambda *d: model(to_device(d[0])))

            dw = random_pvector(lc, device=device)
            dw = step / dw.norm() * dw

            output_before = get_output_vector(loader, function)
            update_model(parameters, dw.get_flat_representation())
            output_after = get_output_vector(loader, function)
            update_model(parameters, -dw.get_flat_representation())

            diff = (((output_before - output_after)**2).sum() /
                    output_before.size(0))

            quot = (diff / F.vTMv(dw)) ** .5

            quots.append(quot.item())

        mean_quotient = sum(quots) / len(quots)
        assert mean_quotient > 1 - 5e-2 and mean_quotient < 1 + 5e-2
