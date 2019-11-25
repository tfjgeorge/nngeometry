from nngeometry.pspace import M2Gradients
from nngeometry.representations import BlockDiagMatrix, KFACMatrix
from nngeometry.vector import random_pvector, PVector
from nngeometry.utils import get_individual_modules
from nngeometry.maths import kronecker
from subsampled_mnist import get_dataset, default_datapath
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from utils import check_ratio, check_tensors
from test_m2gradients import get_fullyconnect_task


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
        return F.log_softmax(out, dim=1)


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
        return F.log_softmax(x, dim=1)


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

    def loss_function(input, target):
        return F.nll_loss(net(input), target, reduction='none')

    return train_loader, net, loss_function


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

    def loss_function(input, target):
        return F.nll_loss(net(input), target, reduction='none')

    return train_loader, net, loss_function


def to_onexdataset(dataset, device):
    # this weird dataset only uses a single input x repeated, it is only
    # designed to test kfac since in this case KFAC and regular Fisher
    # are the same
    loader = torch.utils.data.DataLoader(dataset, len(dataset))
    x, t = next(iter(loader))
    x = x[0, :].repeat(x.size(0), 1)
    return torch.utils.data.TensorDataset(x.to(device), t.to(device))


def test_pspace_blockdiag_vs_kfac():
    """
    Compares blockdiag and kfac representation on datasets/architectures
    where they are the same
    """
    for get_task in [get_convnet_kfc_task, get_fullyconnect_kfac_task]:
        train_loader, net, loss_function = get_task()

        m2_generator = M2Gradients(model=net,
                                   dataloader=train_loader,
                                   loss_function=loss_function)
        M_kfac = KFACMatrix(m2_generator)
        M_blockdiag = BlockDiagMatrix(m2_generator)

        G_kfac = M_kfac.get_matrix(split_weight_bias=True)
        G_blockdiag = M_blockdiag.get_matrix()
        check_tensors(G_blockdiag, G_kfac)

        trace_bd = M_blockdiag.trace()
        trace_kfac = M_kfac.trace()
        check_ratio(trace_bd, trace_kfac)

        # sample random vector
        random_v = random_pvector(net)
        m_norm_kfac = M_kfac.vTMv(random_v)
        m_norm_blockdiag = M_blockdiag.vTMv(random_v)
        check_ratio(m_norm_blockdiag, m_norm_kfac)

        frob_bd = M_blockdiag.frobenius_norm()
        frob_kfac = M_kfac.frobenius_norm()
        check_ratio(frob_bd, frob_kfac)

        check_tensors(M_blockdiag.mv(random_v).get_flat_representation(),
                      M_kfac.mv(random_v).get_flat_representation())


def angle(v1, v2):
    v1_flat = v1.get_flat_representation()
    v2_flat = v2.get_flat_representation()
    return torch.dot(v1_flat, v2_flat) / \
        (torch.norm(v1_flat) * torch.norm(v2_flat))


def test_pspace_kfac_eigendecomposition():
    """
    Check KFAC eigendecomposition by comparing Mv products with v
    where v are the top eigenvectors. The remaining ones can be
    more erratic because of numerical precision
    """
    eps = 1e-3
    train_loader, net, loss_function = get_fullyconnect_task()

    m2_generator = M2Gradients(model=net,
                               dataloader=train_loader,
                               loss_function=loss_function)

    M_kfac = KFACMatrix(m2_generator)
    M_kfac.compute_eigendecomposition()
    evals, evecs = M_kfac.get_eigendecomposition()
    # Loop through all vectors in KFE
    mods, p_pos = get_individual_modules(net)
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
                evec_v = PVector(net, dict_repr=evec_v)
                Mv = M_kfac.mv(evec_v)
                angle_v_Mv = angle(Mv, evec_v)
                assert angle_v_Mv < 1 + eps and angle_v_Mv > 1 - eps
                norm_mv = torch.norm(Mv.get_flat_representation())
                check_ratio(evals[m][0][i_a] * evals[m][1][i_g], norm_mv)
