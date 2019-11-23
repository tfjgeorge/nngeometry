from nngeometry.pspace import M2Gradients
from nngeometry.representations import BlockDiagMatrix, KFACMatrix
from nngeometry.vector import random_pvector
from subsampled_mnist import get_dataset, default_datapath
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from utils import check_ratio, check_tensors


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
