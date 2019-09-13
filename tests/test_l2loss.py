from nngeometry.pspace import L2Loss
from nngeometry.ispace import L2Loss as ISpace_L2Loss
from nngeometry.representations import DenseMatrix, ImplicitMatrix, LowRankMatrix
from subsampled_mnist import get_dataset
import torch
import torch.nn as nn
import torch.nn.functional as tF
from torch.utils.data import DataLoader, Subset

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

def update_model(net, dw):
    # new_net = net.clone()
    i = 0
    for p in net.parameters():
        j = i + p.numel()
        p.data += dw[i:j].view(*p.size())
        i = j

def get_l_vector(dataloader, loss_closure):
    with torch.no_grad():
        l = torch.zeros((len(dataloader.sampler),), device='cuda')
        i = 0
        for inputs, targets in dataloader: 
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            l[i:i+inputs.size(0)] = loss_closure(inputs, targets)
            i += inputs.size(0)
        return l

def get_fullyconnect_task(bs=1000, subs=None):
    train_set = get_dataset('train')
    if subs is not None:
        train_set = Subset(train_set, range(subs))
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=bs,
        shuffle=False)
    net = Net(in_size=10)
    net.to('cuda')
    loss_closure = lambda input, target: tF.nll_loss(net(input), target, reduction='sum')
    return train_loader, net, loss_closure

def test_pspace_l2loss():
    train_loader, net, loss_closure = get_fullyconnect_task()

    el2 = L2Loss(model=net, dataloader=train_loader, loss_closure=loss_closure)
    M = DenseMatrix(el2)

    # compare with || l(w+dw) - l(w) ||_F for randomly sampled dw
    loss_closure = lambda input, target: tF.nll_loss(net(input), target, reduction='none')
    l_0 = get_l_vector(train_loader, loss_closure)
    eps = 1e-3
    dw = torch.rand((M.size(0),), device='cuda')
    dw /= torch.norm(dw)
    update_model(net, eps * dw)
    l_upd = get_l_vector(train_loader, loss_closure)
    update_model(net, -eps * dw)
    ratios = torch.norm(l_upd - l_0)**2 / len(train_loader.sampler) / torch.dot(M.mv(dw), dw) / eps ** 2
    assert ratios < 1.01 and ratios > .99

    # compare project_to_diag to project_from_diag
    M.compute_eigendecomposition()
    assert torch.norm(dw - M.project_to_diag(M.project_from_diag(dw))) < 1e-4

    # project M to its diag space and compare to the evals
    M2 = torch.stack([M.project_to_diag(M.get_matrix()[:, i]) for i in range(M.size(0))])
    M2 = torch.stack([M.project_to_diag(M2[:, i]) for i in range(M.size(0))])
    assert torch.norm(M2 - torch.diag(M.evals)) < 1e-4


def test_pspace_vs_ispace():
    train_loader, net, loss_closure = get_fullyconnect_task()

    ispace_el2 = ISpace_L2Loss(model=net, dataloader=train_loader, loss_closure=loss_closure)
    GM = DenseMatrix(ispace_el2)

    el2 = L2Loss(model=net, dataloader=train_loader, loss_closure=loss_closure)
    M = DenseMatrix(el2)

    n_examples = len(train_loader.sampler)
    ratios_trace = GM.trace() / M.trace() / n_examples
    assert ratios_trace < 1.01 and ratios_trace > .99

def test_pspace_implicit_vs_dense():
    train_loader, net, loss_closure = get_fullyconnect_task()

    el2 = L2Loss(model=net, dataloader=train_loader, loss_closure=loss_closure)
    M_dense = DenseMatrix(el2)
    M_implicit = ImplicitMatrix(el2)

    eps = 1e-3
    dw = torch.rand((M_dense.size(0),), device='cuda')
    dw *= eps / torch.norm(dw)

    M_norm_imp = M_implicit.m_norm(dw)
    M_norm_den = M_dense.m_norm(dw)
    ratio_m_norms = M_norm_imp / M_norm_den
    assert ratio_m_norms < 1.01 and ratio_m_norms > .99

def test_pspace_lowrank_vs_dense():
    train_loader, net, loss_closure = get_fullyconnect_task(bs=100, subs=500)

    el2 = L2Loss(model=net, dataloader=train_loader, loss_closure=loss_closure)
    M_dense = DenseMatrix(el2)
    M_lowrank = LowRankMatrix(el2)

    assert torch.norm(M_dense.get_matrix() - M_lowrank.get_matrix()) < 1e-3

    eps = 1e-3
    dw = torch.rand((M_dense.size(0),), device='cuda')
    dw *= eps / torch.norm(dw)

    M_norm_lr = M_lowrank.m_norm(dw)
    M_norm_den = M_dense.m_norm(dw)
    ratio_m_norms = M_norm_lr / M_norm_den
    assert ratio_m_norms < 1.01 and ratio_m_norms > .99