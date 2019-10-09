from nngeometry.pspace import L2Loss
from nngeometry.representations import BlockDiagMatrix, KFACMatrix
from subsampled_mnist import get_dataset, default_datapath
import torch
import torch.nn as nn
import torch.nn.functional as tF
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

class Net(nn.Module):
    def __init__(self, in_size=10, out_size=10, n_hidden=2, hidden_size=25,
                 nonlinearity=nn.ReLU):
        super(Net, self).__init__()
        layers = []
        sizes = [in_size] + [hidden_size] * n_hidden + [out_size]
        for s_in, s_out in zip(sizes[:-1], sizes[1:]):
            layers.append(nn.Linear(s_in, s_out))
            layers.append(nonlinearity())
        layers.pop() # remove last nonlin
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.net(x)
        return tF.log_softmax(out, dim=1)

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
    loss_closure = lambda input, target: tF.nll_loss(net(input), target, reduction='sum')
    return train_loader, net, loss_closure

def to_onexdataset(dataset, device):
    # this weird dataset only uses a single input x repeated, it is only
    # designed to test kfac since in this case KFAC and regular Fisher
    # are the same
    loader = torch.utils.data.DataLoader(dataset, len(dataset))
    x, t = next(iter(loader))
    x = x[0, :].repeat(x.size(0), 1)
    return torch.utils.data.TensorDataset(x.to(device), t.to(device))

def get_convnet_task(bs=1000, subs=None):
    train_set = datasets.MNIST(default_datapath, train=True, download=True,
                               transform=transforms.ToTensor())
    if subs is not None:
        train_set = Subset(train_set, range(subs))
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=bs,
        shuffle=False)
    net = ConvNet()
    net.to('cuda')
    loss_closure = lambda input, target: tF.nll_loss(net(input), target, reduction='sum')
    return train_loader, net, loss_closure

def test_pspace_blockdiag_vs_kfac():
    for get_task in [get_fullyconnect_kfac_task]:
        train_loader, net, loss_closure = get_task()

        el2 = L2Loss(model=net, dataloader=train_loader, loss_closure=loss_closure)
        M_kfac = KFACMatrix(el2)
        M_blockdiag = BlockDiagMatrix(el2)

        eps = 1e-3
        G_kfac = M_kfac.get_matrix(split_weight_bias=True)
        G_blockdiag = M_blockdiag.get_matrix()
        assert torch.norm(G_kfac - G_blockdiag) < eps

        trace_bd = M_blockdiag.trace()
        trace_kfac = M_kfac.trace()
        ratio_trace = trace_bd / trace_kfac
        assert ratio_trace < 1.01 and ratio_trace > .99
