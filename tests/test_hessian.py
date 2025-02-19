# %%
import pytest
import torch
from nngeometry import Hessian, FIM
from nngeometry.object.pspace import PMatDense
from tasks import (
    get_linear_fc_task,
    get_linear_conv_task,
    get_batchnorm_fc_linear_task,
    get_batchnorm_conv_linear_task,
    get_fullyconnect_onlylast_task,
    to_device,
)
from utils import check_tensors

linear_tasks = [
    get_linear_fc_task,
    get_linear_conv_task,
    get_fullyconnect_onlylast_task,
]


@pytest.fixture(autouse=True)
def make_test_deterministic():
    torch.manual_seed(1234)
    yield


def test_hessian_vs_FIM():
    for get_task in linear_tasks:

        print(get_task)
        loader, lc, parameters, model, _ = get_task()
        model.train()

        F = FIM(
            layer_collection=lc,
            model=model,
            loader=loader,
            variant="classif_logits",
            representation=PMatDense,
            function=lambda *d: model(to_device(d[0])),
        )

        def f(y_pred, y):
            return torch.nn.functional.cross_entropy(y_pred, y, reduction="sum")

        H = Hessian(
            layer_collection=lc,
            model=model,
            loader=loader,
            representation=PMatDense,
            function=f,
        )

        check_tensors(F.get_dense_tensor(), H.get_dense_tensor(), only_print_diff=True)
        check_tensors(F.get_dense_tensor(), H.get_dense_tensor())


# # # %%

# import matplotlib.pyplot as plt


# def p(M):
#     mm = torch.abs(M).max()
#     plt.figure()
#     plt.matshow(M, cmap="PiYG", vmin=-mm, vmax=mm)
#     plt.show()


# # %%
# linear_tasks = [get_linear_conv_task] * 1
# for i, get_task in enumerate(linear_tasks):

#     torch.manual_seed(48)
#     print("-----------------")
#     print(i, get_task)
#     loader, lc, parameters, model, function = get_task()
#     model.eval()

#     F = FIM(
#         layer_collection=lc,
#         model=model,
#         loader=loader,
#         variant="classif_logits",
#         representation=PMatDense,
#         function=lambda *d: model(to_device(d[0])),
#     )

#     def f(y_pred, y):
#         # print(y_pred.size(), y.size())
#         return torch.nn.functional.cross_entropy(y_pred, y, reduction="sum")

#     H = Hessian(
#         layer_collection=lc,
#         model=model,
#         loader=loader,
#         representation=PMatDense,
#         function=f,
#     )

#     F_flat = F.get_dense_tensor()
#     H_flat = H.get_dense_tensor().detach()

#     p(F_flat)

#     p(H_flat)

#     if (
#         check_tensors(F.get_dense_tensor(), H.get_dense_tensor(), only_print_diff=True)
#         > 0.01
#     ):

#         p(F_flat - H_flat)


# # %%
# lc.layers["conv1"].numel(), lc.layers["conv2"].numel()
# # %%
# lc.layers
# # %%
# H2_flat = Hessian(
#     layer_collection=lc,
#     model=model,
#     loader=loader,
#     representation=PMatDense,
#     function=f,
# ).get_dense_tensor()
# p(H2_flat)

# # %%
# F2_flat = FIM(
#     layer_collection=lc,
#     model=model,
#     loader=loader,
#     variant="classif_logits",
#     representation=PMatDense,
#     function=lambda *d: model(to_device(d[0])),
# ).get_dense_tensor()
# p(F2_flat)

# # %%
