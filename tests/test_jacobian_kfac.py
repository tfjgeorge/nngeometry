from nngeometry.generator.jacobian import Jacobian
from nngeometry.object.pspace import PMatBlockDiag, PMatKFAC
from nngeometry.object.vector import random_pvector, PVector
from nngeometry.maths import kronecker
from nngeometry.layercollection import LayerCollection
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from utils import check_ratio, check_tensors, angle
from tasks import get_fullyconnect_task, get_mnist, get_conv_task, to_device_model
import os
import pytest

default_datapath = 'tmp'
if 'SLURM_TMPDIR' in os.environ:
    default_datapath = os.path.join(os.environ['SLURM_TMPDIR'], 'data')

if torch.cuda.is_available():
    device = 'cuda'

    def to_device(tensor):
        return tensor.to(device)
else:
    device = 'cpu'

    # on cpu we need to use double as otherwise ill-conditioning in sums
    # causes numerical instability
    def to_device(tensor):
        return tensor.double()


class Net(nn.Module):
    def __init__(self, in_size=10, out_size=10, n_hidden=2, hidden_size=25,
                 nonlinearity=nn.ReLU):
        super(Net, self).__init__()
        layers = []
        sizes = [in_size] + [hidden_size] * n_hidden + [out_size]
        for s_in, s_out in zip(sizes[:-1], sizes[1:]):
            layers.append(nn.Linear(s_in, s_out))
            layers.append(nonlinearity())
        # remove last nonlinearity
        layers.pop()
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.net(x)
        return out


class ConvNet(nn.Module):
    # this weird network transforms the input so that KFC and regular
    # block diagonal Fisher are the same
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, 3, 1)
        self.conv2 = nn.Conv2d(5, 6, 4, 1)
        self.conv3 = nn.Conv2d(6, 7, 3, 1)
        self.fc1 = nn.Linear(1*1*7, 10)

    def forward(self, x):
        # TODO fix this (backprop gradient is 0)
        x = torch.ones_like(x) * 0.1 + x - x
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = torch.ones_like(x) * 0.2 + x - x
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = torch.ones_like(x) * 0.3 + x - x
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 1*1*7)
        x = x[0, :].repeat(x.size(0), 1)
        x = self.fc1(x)
        return x


def get_fullyconnect_kfac_task(bs=300):
    train_set = get_mnist()
    train_set = Subset(train_set, range(1000))
    train_set = to_onexdataset(train_set, device)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=bs,
        shuffle=False)

    net = Net(in_size=18*18)
    to_device_model(net)
    net.eval()

    def output_fn(input, target):
        return net(to_device(input))

    layer_collection = LayerCollection.from_model(net)
    return (train_loader, layer_collection, net.parameters(), net,
            output_fn, 10)


def to_onexdataset(dataset, device):
    # this weird dataset only uses a single input x repeated, it is only
    # designed to test kfac since in this case KFAC and regular Fisher
    # are the same
    loader = torch.utils.data.DataLoader(dataset, len(dataset))
    x, t = next(iter(loader))
    x = x[0, :, 5:-5, 5:-5].contiguous().view(1, -1).repeat(x.size(0), 1)
    return torch.utils.data.TensorDataset(x.to(device), t.to(device))


def get_convnet_kfc_task(bs=300):
    train_set = get_mnist()
    train_set = Subset(train_set, range(1000))
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=bs,
        shuffle=False)
    net = ConvNet()
    to_device_model(net)
    net.eval()

    def output_fn(input, target):
        return net(to_device(input))

    layer_collection = LayerCollection.from_model(net)
    return (train_loader, layer_collection, net.parameters(), net,
            output_fn, 10)


@pytest.fixture(autouse=True)
def make_test_deterministic():
    torch.manual_seed(1234)
    yield


def test_jacobian_kfac_vs_pblockdiag():
    """
    Compares blockdiag and kfac representation on datasets/architectures
    where they are the same

    TODO: design a task where kfc is exact
    """
    # for get_task in [get_convnet_kfc_task, get_fullyconnect_kfac_task]:
    for get_task in [get_fullyconnect_kfac_task]:
        loader, lc, parameters, model, function, n_output = get_task()

        generator = Jacobian(layer_collection=lc,
                             model=model,
                             function=function,
                             n_output=n_output)
        M_kfac = PMatKFAC(generator=generator, examples=loader)
        M_blockdiag = PMatBlockDiag(generator=generator, examples=loader)

        G_kfac = M_kfac.get_dense_tensor(split_weight_bias=True)
        G_blockdiag = M_blockdiag.get_dense_tensor()
        check_tensors(G_blockdiag, G_kfac, only_print_diff=True)


def test_jacobian_kfac():
    for get_task in [get_fullyconnect_task, get_conv_task]:
        loader, lc, parameters, model, function, n_output = get_task()

        generator = Jacobian(layer_collection=lc,
                             model=model,
                             function=function,
                             n_output=n_output)
        M_kfac = PMatKFAC(generator=generator, examples=loader)
        G_kfac_split = M_kfac.get_dense_tensor(split_weight_bias=True)
        G_kfac = M_kfac.get_dense_tensor(split_weight_bias=False)

        # Test trace
        trace_direct = torch.trace(G_kfac_split)
        trace_kfac = M_kfac.trace()
        check_ratio(trace_direct, trace_kfac)

        # Test frobenius norm
        frob_direct = torch.norm(G_kfac)
        frob_kfac = M_kfac.frobenius_norm()
        check_ratio(frob_direct, frob_kfac)

        # Test get_diag
        check_tensors(torch.diag(G_kfac_split),
                      M_kfac.get_diag(split_weight_bias=True))

        # sample random vector
        random_v = random_pvector(lc, device)

        # Test mv
        mv_direct = torch.mv(G_kfac_split, random_v.get_flat_representation())
        mv_kfac = M_kfac.mv(random_v)
        check_tensors(mv_direct,
                      mv_kfac.get_flat_representation())

        # Test vTMv
        mnorm_kfac = M_kfac.vTMv(random_v)
        mnorm_direct = torch.dot(mv_direct, random_v.get_flat_representation())
        check_ratio(mnorm_direct, mnorm_kfac)

        # Test inverse
        # We start from a mv vector since it kills its components projected to
        # the small eigenvalues of KFAC
        regul = 1e-7

        mv2 = M_kfac.mv(mv_kfac)
        kfac_inverse = M_kfac.inverse(regul)
        mv_back = kfac_inverse.mv(mv2 + regul * mv_kfac)
        check_tensors(mv_kfac.get_flat_representation(),
                      mv_back.get_flat_representation(),
                      eps=1e-2)

        # Test solve
        mv_back = M_kfac.solve(mv2 + regul * mv_kfac, regul=regul)
        check_tensors(mv_kfac.get_flat_representation(),
                      mv_back.get_flat_representation(),
                      eps=1e-2)


def test_pspace_kfac_eigendecomposition():
    """
    Check KFAC eigendecomposition by comparing Mv products with v
    where v are the top eigenvectors. The remaining ones can be
    more erratic because of numerical precision
    """
    eps = 1e-3
    loader, lc, parameters, model, function, n_output = get_fullyconnect_task()

    generator = Jacobian(layer_collection=lc,
                         model=model,
                         function=function,
                         n_output=n_output)

    M_kfac = PMatKFAC(generator=generator, examples=loader)
    M_kfac.compute_eigendecomposition()
    evals, evecs = M_kfac.get_eigendecomposition()
    # Loop through all vectors in KFE
    l_to_m, _ = lc.get_layerid_module_maps(model)
    for l_id, layer in lc.layers.items():
        for i_a in range(-3, 0):
            for i_g in range(-3, 0):
                evec_v = dict()
                for l_id2, layer2 in lc.layers.items():
                    m = l_to_m[l_id2]
                    if l_id2 == l_id:
                        v_a = evecs[l_id][0][:, i_a].unsqueeze(0)
                        v_g = evecs[l_id][1][:, i_g].unsqueeze(1)
                        evec_block = kronecker(v_g, v_a)
                        evec_tuple = (evec_block[:, :-1].contiguous(),
                                      evec_block[:, -1].contiguous())
                        evec_v[l_id] = evec_tuple
                    else:
                        evec_v[l_id2] = (torch.zeros_like(m.weight),
                                         torch.zeros_like(m.bias))
                evec_v = PVector(lc, dict_repr=evec_v)
                Mv = M_kfac.mv(evec_v)
                angle_v_Mv = angle(Mv, evec_v)
                assert angle_v_Mv < 1 + eps and angle_v_Mv > 1 - eps
                norm_mv = torch.norm(Mv.get_flat_representation())
                check_ratio(evals[l_id][0][i_a] * evals[l_id][1][i_g], norm_mv)


def test_kfac():
    for get_task in [get_fullyconnect_task, get_conv_task]:
        loader, lc, parameters, model1, function1, n_output = get_task()
        _, _, _, model2, function2, _ = get_task()

        generator1 = Jacobian(layer_collection=lc,
                              model=model1,
                              function=function1,
                              n_output=n_output)
        generator2 = Jacobian(layer_collection=lc,
                              model=model2,
                              function=function1,
                              n_output=n_output)
        M_kfac1 = PMatKFAC(generator=generator1, examples=loader)
        M_kfac2 = PMatKFAC(generator=generator2, examples=loader)

        prod = M_kfac1.mm(M_kfac2)

        M_kfac1_tensor = M_kfac1.get_dense_tensor(split_weight_bias=True)
        M_kfac2_tensor = M_kfac2.get_dense_tensor(split_weight_bias=True)

        prod_tensor = prod.get_dense_tensor(split_weight_bias=True)

        check_tensors(torch.mm(M_kfac1_tensor, M_kfac2_tensor),
                      prod_tensor)