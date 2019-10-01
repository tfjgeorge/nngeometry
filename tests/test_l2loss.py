from nngeometry.pspace import L2Loss
from nngeometry.ispace import L2Loss as ISpace_L2Loss
from nngeometry.representations import DenseMatrix, ImplicitMatrix, LowRankMatrix, DiagMatrix
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

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, 3, 1)
        self.conv2 = nn.Conv2d(5, 6, 4, 1)
        self.conv3 = nn.Conv2d(6, 7, 3, 1)
        self.fc1 = nn.Linear(1*1*7, 10)

    def forward(self, x):
        x = tF.relu(self.conv1(x))
        x = tF.max_pool2d(x, 2, 2)
        x = tF.relu(self.conv2(x))
        x = tF.max_pool2d(x, 2, 2)
        x = tF.relu(self.conv3(x))
        x = tF.max_pool2d(x, 2, 2)
        x = x.view(-1, 1*1*7)
        x = self.fc1(x)
        return tF.log_softmax(x, dim=1)

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

def test_pspace_l2loss():
    for get_task in [get_convnet_task, get_fullyconnect_task]:
        train_loader, net, loss_closure = get_task()

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

        for impl in ['symeig', 'svd']:
            # compare project_to_diag to project_from_diag
            M.compute_eigendecomposition(impl)
            assert torch.norm(dw - M.project_to_diag(M.project_from_diag(dw))) < 1e-4

            # project M to its diag space and compare to the evals
            M2 = torch.stack([M.project_to_diag(M.get_matrix()[:, i]) for i in range(M.size(0))])
            M2 = torch.stack([M.project_to_diag(M2[:, i]) for i in range(M.size(0))])
            assert torch.norm(M2 - torch.diag(M.evals)) < 1e-4

            # same but directly with the matrix:
            assert torch.norm(M.project_to_diag(M.get_matrix()) - torch.diag(M.evals)) < 1e-4

            evals, evecs = M.get_eigendecomposition()
            assert torch.norm(torch.mm(torch.mm(evecs, torch.diag(evals)), evecs.t()) - M.get_matrix()) < 1e-3

        # compare frobenius norm to trace(M^T M)
        f_norm = M.frobenius_norm()
        f_norm2 = torch.trace(torch.mm(M.get_matrix().t(), M.get_matrix()))**.5
        ratio = f_norm / f_norm2
        assert ratio < 1.01 and ratio > .99

def test_pspace_vs_ispace():
    train_loader, net, loss_closure = get_fullyconnect_task()

    ispace_el2 = ISpace_L2Loss(model=net, dataloader=train_loader, loss_closure=loss_closure)
    MIspace = DenseMatrix(ispace_el2)

    el2 = L2Loss(model=net, dataloader=train_loader, loss_closure=loss_closure)
    M = DenseMatrix(el2)

    n_examples = len(train_loader.sampler)
    ratios_trace = MIspace.trace() / M.trace() / n_examples
    assert ratios_trace < 1.01 and ratios_trace > .99

    MIspace_frob = MIspace.frobenius_norm()
    M_frob = M.frobenius_norm()
    ratios_frob = MIspace_frob / M_frob / n_examples
    assert ratios_frob < 1.01 and ratios_frob > .99

def test_pspace_implicit_vs_dense():
    for get_task in [get_convnet_task, get_fullyconnect_task]:
        train_loader, net, loss_closure = get_task()

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

        trace_imp = M_implicit.trace()
        trace_den = M_dense.trace()
        ratio_trace = trace_imp / trace_den
        assert ratio_trace < 1.01 and ratio_trace > .99

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

    assert torch.norm(M_dense.mv(dw) - M_lowrank.mv(dw)) < 1e-3

def test_pspace_lowrank():
    for get_task in [get_convnet_task, get_fullyconnect_task]:
        train_loader, net, loss_closure = get_fullyconnect_task(bs=100, subs=500)
        el2 = L2Loss(model=net, dataloader=train_loader, loss_closure=loss_closure)
        M = LowRankMatrix(el2)

        M.compute_eigendecomposition()

        evals, evecs = M.get_eigendecomposition()
        assert torch.norm(torch.mm(torch.mm(evecs, torch.diag(evals)), evecs.t()) - M.get_matrix()) < 1e-3

        assert torch.norm(M.project_to_diag(M.get_matrix()) - torch.diag(M.evals)) < 1e-3

        # check evecs:
        for i in range(evecs.size(1)):
            v = evecs[:, i]
            norm_v = torch.norm(v)
            if not (norm_v > 0.999 and norm_v < 1.001):
                # TODO improve this
                print(i, norm_v)
            assert torch.norm(M.mv(v) - v * evals[i]) < 1e-3

def test_pspace_diag_vs_dense():
    train_loader, net, loss_closure = get_fullyconnect_task(bs=100, subs=500)

    el2 = L2Loss(model=net, dataloader=train_loader, loss_closure=loss_closure)
    M_dense = DenseMatrix(el2)
    M_diag = DiagMatrix(el2)

    assert torch.norm(torch.diag(M_dense.get_matrix() - M_diag.get_matrix())) < 1e-3

    trace_diag = M_diag.trace()
    trace_den = M_dense.trace()
    ratio_trace = trace_diag / trace_den
    assert ratio_trace < 1.01 and ratio_trace > .99

    eps = 1e-3
    dw = torch.rand((M_dense.size(0),), device='cuda')
    assert torch.norm(M_diag.mv(dw) - 
                      torch.mv(torch.diag(torch.diag(M_dense.get_matrix())), dw)) < 1e-3

def test_ispace_dense_vs_implicit():
    train_loader, net, loss_closure = get_fullyconnect_task()

    ispace_el2 = ISpace_L2Loss(model=net, dataloader=train_loader, loss_closure=loss_closure)
    M_dense = DenseMatrix(ispace_el2)
    M_implicit = ImplicitMatrix(ispace_el2)

    n_examples = len(train_loader.sampler)
    v = torch.rand((n_examples,), device='cuda')

    m_norm_dense = M_dense.m_norm(v)
    m_norm_implicit = M_implicit.m_norm(v)

    ratios_norms = m_norm_dense / m_norm_implicit
    assert ratios_norms < 1.01 and ratios_norms > .99

    frob_norm_dense = M_dense.frobenius_norm()
    frob_norm_implicit = M_implicit.frobenius_norm()
    print(frob_norm_dense, frob_norm_implicit)