import pytest
import torch
import torch.nn.functional as tF
from tasks import (
    device,
    get_conv_gn_task,
    get_conv_task,
    get_fullyconnect_segm_task,
    get_fullyconnect_task,
    to_device,
)
from utils import get_output_vector, update_model

from nngeometry.metrics import FIM, FIM_MonteCarlo
from nngeometry.object.pspace import PMatDense
from nngeometry.object.vector import random_pvector

nonlinear_tasks = [get_conv_gn_task, get_fullyconnect_task, get_conv_task]


@pytest.fixture(autouse=True)
def make_test_deterministic():
    torch.manual_seed(1234)
    yield


def test_FIM_MC_vs_linearization():
    step = 1e-2

    for get_task in nonlinear_tasks:
        for variant in ["classif_logits", "classif_logsoftmax"]:
            quots = []
            for i in range(10):  # repeat to kill statistical fluctuations
                loader, lc, parameters, model, function = get_task()
                model.train()

                if variant == "classif_logits":
                    f = lambda *d: model(to_device(d[0]))
                elif variant == "classif_logsoftmax":
                    f = lambda *d: torch.log_softmax(model(to_device(d[0])), dim=1)

                F = FIM_MonteCarlo(
                    layer_collection=lc,
                    model=model,
                    loader=loader,
                    variant=variant,
                    representation=PMatDense,
                    trials=10,
                    function=f,
                )

                dw = random_pvector(lc, device=device)
                dw = step / dw.norm() * dw

                output_before = get_output_vector(loader, function)
                update_model(parameters, dw.to_torch())
                output_after = get_output_vector(loader, function)
                update_model(parameters, -dw.to_torch())

                KL = tF.kl_div(
                    tF.log_softmax(output_before, dim=1),
                    tF.log_softmax(output_after, dim=1),
                    log_target=True,
                    reduction="batchmean",
                )

                quot = (KL / F.vTMv(dw) * 2) ** 0.5

                quots.append(quot.item())

            mean_quotient = sum(quots) / len(quots)
            assert mean_quotient > 1 - 5e-2 and mean_quotient < 1 + 5e-2


def test_FIM_vs_linearization_classif_logits():
    step = 1e-2

    for get_task in nonlinear_tasks:
        quots = []
        for i in range(10):  # repeat to kill statistical fluctuations
            loader, lc, parameters, model, function = get_task()
            model.train()
            F = FIM(
                layer_collection=lc,
                model=model,
                loader=loader,
                variant="classif_logits",
                representation=PMatDense,
                function=lambda *d: model(to_device(d[0])),
            )

            dw = random_pvector(lc, device=device)
            dw = step / dw.norm() * dw

            output_before = get_output_vector(loader, function)
            update_model(parameters, dw.to_torch())
            output_after = get_output_vector(loader, function)
            update_model(parameters, -dw.to_torch())

            KL = tF.kl_div(
                tF.log_softmax(output_before, dim=1),
                tF.log_softmax(output_after, dim=1),
                log_target=True,
                reduction="batchmean",
            )

            quot = (KL / F.vTMv(dw) * 2) ** 0.5

            quots.append(quot.item())

        mean_quotient = sum(quots) / len(quots)

        assert mean_quotient > 1 - 1e-3 and mean_quotient < 1 + 1e-3


def test_FIM_vs_linearization_classif_binary_logits():
    step = 1e-3

    for get_task in nonlinear_tasks:
        loader, lc, parameters, model, function = get_task(binary=True)
        model.train()
        F = FIM(
            layer_collection=lc,
            model=model,
            loader=loader,
            variant="classif_binary_logits",
            representation=PMatDense,
            function=lambda *d: model(to_device(d[0])),
        )

        dw = random_pvector(lc, device=device)
        dw = step / dw.norm() * dw

        logits_before = get_output_vector(loader, function)
        update_model(parameters, dw.to_torch())
        logits_after = get_output_vector(loader, function)
        update_model(parameters, -dw.to_torch())

        log_prob_1_before = tF.logsigmoid(logits_before)
        log_prob_0_before = tF.logsigmoid(-logits_before)
        log_prob_1_after = tF.logsigmoid(logits_after)
        log_prob_0_after = tF.logsigmoid(-logits_after)

        KL = tF.kl_div(
            torch.cat((log_prob_1_before, log_prob_0_before), dim=1),
            torch.cat((log_prob_1_after, log_prob_0_after), dim=1),
            log_target=True,
            reduction="batchmean",
        )

        quot = (KL / F.vTMv(dw) * 2) ** 0.5

        tol = 1e-3
        assert quot > 1 - tol and quot < 1 + tol


def test_FIM_vs_linearization_regression():
    step = 1e-2

    for get_task in nonlinear_tasks:
        quots = []
        for i in range(10):  # repeat to kill statistical fluctuations
            loader, lc, parameters, model, function = get_task()
            model.train()
            F = FIM(
                layer_collection=lc,
                model=model,
                loader=loader,
                variant="regression",
                representation=PMatDense,
                function=lambda *d: model(to_device(d[0])),
            )

            dw = random_pvector(lc, device=device)
            dw = step / dw.norm() * dw

            output_before = get_output_vector(loader, function)
            update_model(parameters, dw.to_torch())
            output_after = get_output_vector(loader, function)
            update_model(parameters, -dw.to_torch())

            diff = ((output_before - output_after) ** 2).sum() / output_before.size(0)

            quot = (diff / F.vTMv(dw)) ** 0.5

            quots.append(quot.item())

        mean_quotient = sum(quots) / len(quots)
        assert mean_quotient > 1 - 5e-2 and mean_quotient < 1 + 5e-2


def test_FIM_MC_vs_linearization_segmentation():
    step = 1e-2
    variant = "segmentation_logits"
    for get_task in [get_fullyconnect_segm_task]:
        quots = []
        for i in range(10):  # repeat to kill statistical fluctuations
            loader, lc, parameters, model, function = get_task()
            model.train()

            f = lambda *d: model(to_device(d[0]))

            F = FIM_MonteCarlo(
                layer_collection=lc,
                model=model,
                loader=loader,
                variant=variant,
                representation=PMatDense,
                trials=10,
                function=f,
            )

            dw = random_pvector(lc, device=device)
            dw = step / dw.norm() * dw

            output_before = get_output_vector(loader, function)
            update_model(parameters, dw.to_torch())
            output_after = get_output_vector(loader, function)
            update_model(parameters, -dw.to_torch())

            KL = tF.kl_div(
                tF.log_softmax(output_before, dim=1),
                tF.log_softmax(output_after, dim=1),
                log_target=True,
                reduction="batchmean",
            )

            quot = (KL / F.vTMv(dw) * 2) ** 0.5

            quots.append(quot.item())

        mean_quotient = sum(quots) / len(quots)
        assert mean_quotient > 1 - 5e-2 and mean_quotient < 1 + 5e-2
