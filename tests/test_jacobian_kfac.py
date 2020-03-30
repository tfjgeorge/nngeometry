from nngeometry.generator.jacobian import Jacobian
from nngeometry.object.pspace import PSpaceBlockDiag, PSpaceKFAC
from nngeometry.object.vector import random_pvector, PVector
from nngeometry.utils import get_individual_modules
from nngeometry.maths import kronecker
from nngeometry.layercollection import LayerCollection
from subsampled_mnist import get_dataset, default_datapath
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from utils import check_ratio, check_tensors, angle
from tasks import get_fullyconnect_task


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


def get_fullyconnect_kfac_task(bs=1000, subs=None):
    train_set = get_dataset('train')
    if subs is not None:
        train_set = Subset(train_set, range(subs))
    train_set = to_onexdataset(train_set, 'cuda')
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=bs,
        shuffle=False)

    net = Net(in_size=10)
    net.to('cuda')

    def output_fn(input, target):
        input = input.to('cuda')
        return net(input)

    layer_collection = LayerCollection.from_model(net)
    return (train_loader, layer_collection, net.parameters(), net,
            output_fn, 10)


def get_convnet_kfc_task(bs=1000, subs=None):
    train_set = Subset(datasets.MNIST(root=default_datapath,
                                      train=True,
                                      download=True,
                                      transform=transforms.ToTensor()),
                       range(40000))
    if subs is not None:
        train_set = Subset(train_set, range(subs))
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=bs,
        shuffle=False)
    net = ConvNet()
    net.to('cuda')

    def output_fn(input, target):
        input = input.to('cuda')
        return net(input)

    layer_collection = LayerCollection.from_model(net)
    return (train_loader, layer_collection, net.parameters(), net,
            output_fn, 10)


def to_onexdataset(dataset, device):
    # this weird dataset only uses a single input x repeated, it is only
    # designed to test kfac since in this case KFAC and regular Fisher
    # are the same
    loader = torch.utils.data.DataLoader(dataset, len(dataset))
    x, t = next(iter(loader))
    x = x[0, :].repeat(x.size(0), 1)
    return torch.utils.data.TensorDataset(x.to(device), t.to(device))


def test_jacobian_kfac_vs_pblockdiag():
    """
    Compares blockdiag and kfac representation on datasets/architectures
    where they are the same
    """
    for get_task in [get_fullyconnect_kfac_task]:
        loader, lc, parameters, model, function, n_output = get_task()

        generator = Jacobian(layer_collection=lc,
                             model=model,
                             loader=loader,
                             function=function,
                             n_output=n_output)
        M_kfac = PSpaceKFAC(generator)
        M_blockdiag = PSpaceBlockDiag(generator)

        G_kfac = M_kfac.get_dense_tensor(split_weight_bias=True)
        G_blockdiag = M_blockdiag.get_dense_tensor()
        check_tensors(G_blockdiag, G_kfac)

        trace_bd = M_blockdiag.trace()
        trace_kfac = M_kfac.trace()
        check_ratio(trace_bd, trace_kfac)

        # sample random vector
        random_v = random_pvector(lc, 'cuda')
        m_norm_kfac = M_kfac.vTMv(random_v)
        m_norm_blockdiag = M_blockdiag.vTMv(random_v)
        check_ratio(m_norm_blockdiag, m_norm_kfac)

        frob_bd = M_blockdiag.frobenius_norm()
        frob_kfac = M_kfac.frobenius_norm()
        check_ratio(frob_bd, frob_kfac)

        check_tensors(M_blockdiag.mv(random_v).get_flat_representation(),
                      M_kfac.mv(random_v).get_flat_representation())


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
                         loader=loader,
                         function=function,
                         n_output=n_output)

    M_kfac = PSpaceKFAC(generator)
    M_kfac.compute_eigendecomposition()
    evals, evecs = M_kfac.get_eigendecomposition()
    # Loop through all vectors in KFE
    mods, p_pos = get_individual_modules(model)
    for m in mods:
        for i_a in range(-4, 0):
            for i_g in range(-4, 0):
                evec_v = dict()
                for m2 in mods:
                    if m2 is m:
                        v_a = evecs[m][0][:, i_a].unsqueeze(0)
                        v_g = evecs[m][1][:, i_g].unsqueeze(1)
                        evec_block = kronecker(v_g, v_a)
                        evec_tuple = (evec_block[:, :-1].contiguous(),
                                      evec_block[:, -1].contiguous())
                        evec_v[m2] = evec_tuple
                    else:
                        evec_v[m2] = (torch.zeros_like(m2.weight),
                                      torch.zeros_like(m2.bias))
                evec_v = PVector(lc, dict_repr=evec_v)
                Mv = M_kfac.mv(evec_v)
                angle_v_Mv = angle(Mv, evec_v)
                assert angle_v_Mv < 1 + eps and angle_v_Mv > 1 - eps
                norm_mv = torch.norm(Mv.get_flat_representation())
                check_ratio(evals[m][0][i_a] * evals[m][1][i_g], norm_mv)
