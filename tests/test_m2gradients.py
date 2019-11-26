from nngeometry.pspace import M2Gradients
from nngeometry.ispace import M2Gradients as ISpace_M2Gradients
from nngeometry.representations import (DenseMatrix, ImplicitMatrix,
                                        LowRankMatrix, DiagMatrix,
                                        BlockDiagMatrix)
from nngeometry.vector import PVector, random_pvector, IVector
from nngeometry.utils import get_individual_modules
from subsampled_mnist import get_dataset, default_datapath
import torch
import torch.nn as nn
import torch.nn.functional as tF
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from utils import check_ratio, check_tensors


class Net(nn.Module):
    def __init__(self, in_size=10, out_size=10, n_hidden=2, hidden_size=25,
                 nonlinearity=nn.ReLU, batch_norm=False):
        super(Net, self).__init__()
        layers = []
        sizes = [in_size] + [hidden_size] * n_hidden + [out_size]
        for s_in, s_out in zip(sizes[:-1], sizes[1:]):
            layers.append(nn.Linear(s_in, s_out))
            if batch_norm:
                layers.append(nn.BatchNorm1d(s_out))
            layers.append(nonlinearity())
        # remove last nonlinearity:
        layers.pop()
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
    i = 0
    for p in net.parameters():
        j = i + p.numel()
        p.data += dw[i:j].view(*p.size())
        i = j


def get_l_vector(dataloader, loss_function):
    with torch.no_grad():
        losses = torch.zeros((len(dataloader.sampler),), device='cuda')
        i = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            losses[i:i+inputs.size(0)] = loss_function(inputs, targets)
            i += inputs.size(0)
        return losses


def get_fullyconnect_task(bs=1000, subs=None, batch_norm=False):
    train_set = get_dataset('train')
    if subs is not None:
        train_set = Subset(train_set, range(subs))
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=bs,
        shuffle=False)
    net = Net(in_size=10, batch_norm=batch_norm)
    net.to('cuda')

    def loss_function(input, target):
        return tF.nll_loss(net(input), target, reduction='none')

    return train_loader, net, loss_function


def get_fullyconnect_batchnorm_task(*args, **kwargs):
    return get_fullyconnect_task(*args, batch_norm=True, **kwargs)


def get_convnet_task(bs=1000, subs=None):
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
        return tF.nll_loss(net(input), target, reduction='none')

    return train_loader, net, loss_function


def test_pspace_m2gradients_vs_loss():
    """
    Test || l(w+dw) - l(w) ||_2 against it linerarization given
    by the geometry matrix m2 of the gradients of the loss
    """
    for get_task in [get_convnet_task, get_fullyconnect_task]:
        train_loader, net, loss_function = get_task()

        m2_generator = M2Gradients(model=net,
                                   dataloader=train_loader,
                                   loss_function=loss_function)
        M = DenseMatrix(m2_generator)

        # compare with || l(w+dw) - l(w) ||_F for randomly sampled dw
        l_0 = get_l_vector(train_loader, loss_function)
        eps = 1e-4

        dw = torch.rand((M.size(0),), device='cuda')
        dw *= eps / torch.norm(dw)
        dw_vec = PVector(net, vector_repr=dw)
        update_model(net, dw)
        l_upd = get_l_vector(train_loader, loss_function)
        update_model(net, - dw)
        check_ratio(torch.norm(l_upd - l_0)**2 / len(train_loader.sampler),
                    torch.dot(M.mv(dw_vec).get_flat_representation(), dw))


def test_pspace_m2gradients_bn_vs_loss():
    """
    Test || l(w+dw) - l(w) ||_2 against it linerarization given
    by the geometry matrix m2 of the gradients of the loss
    NB: This is a test for batch norm, notice the following things:
     - the net is put in eval mode
     - the dataset contains a single 1000 examples minibatch
     - in check_ratio in the end we have a much larger value for eps
    than usually
    """
    for get_task in [get_fullyconnect_batchnorm_task]:
        train_loader, net, loss_function = get_task(bs=1000, subs=1000)
        net.eval()

        m2_generator = M2Gradients(model=net,
                                   dataloader=train_loader,
                                   loss_function=loss_function)
        M = DenseMatrix(m2_generator)

        # compare with || l(w+dw) - l(w) ||_F for randomly sampled dw
        l_0 = get_l_vector(train_loader, loss_function)
        eps = 1e-4

        dw = torch.rand((M.size(0),), device='cuda')
        dw *= eps / torch.norm(dw)
        dw_vec = PVector(net, vector_repr=dw)
        update_model(net, dw)
        l_upd = get_l_vector(train_loader, loss_function)
        update_model(net, - dw)
        check_ratio(torch.norm(l_upd - l_0)**2 / len(train_loader.sampler),
                    torch.dot(M.mv(dw_vec).get_flat_representation(), dw),
                    eps=1e-2)


def test_pspace_m2gradients_dense():
    for get_task in [get_convnet_task, get_fullyconnect_task]:
        train_loader, net, loss_function = get_task()

        m2_generator = M2Gradients(model=net,
                                   dataloader=train_loader,
                                   loss_function=loss_function)
        M = DenseMatrix(m2_generator)
        dw = random_pvector(net)
        for impl in ['symeig', 'svd']:
            # compare project_to_diag to project_from_diag
            M.compute_eigendecomposition(impl)
            dw2 = M.project_to_diag(M.project_from_diag(dw))
            check_tensors(dw.get_flat_representation(),
                          dw2.get_flat_representation())

            # project M to its diag space and compare to the evals
            cols = []
            for i in range(M.size(0)):
                vi = PVector(net, vector_repr=M.get_matrix()[:, i])
                cols.append(M.project_to_diag(vi))
            M2 = torch.stack([c.get_flat_representation() for c in cols])
            cols2 = []
            for i in range(M.size(0)):
                vi = PVector(net, vector_repr=M2[:, i])
                cols2.append(M.project_to_diag(vi))
            M2 = torch.stack([c.get_flat_representation() for c in cols2])
            # need to raise eps because of numerical stability
            check_tensors(M2, torch.diag(M.evals), eps=1e-4)

            evals, evecs = M.get_eigendecomposition()
            check_tensors(M.get_matrix(),
                          torch.mm(torch.mm(evecs, torch.diag(evals)),
                                   evecs.t()), eps=1e-4)

        # compare frobenius norm to trace(M^T M)
        f_norm = M.frobenius_norm()
        f_norm2 = torch.trace(torch.mm(M.get_matrix().t(), M.get_matrix()))**.5
        check_ratio(f_norm, f_norm2)


def test_pspace_vs_ispace():
    """
    Check that ISpace Gram matrix and PSpace second moment matrix
    have the same trace and frobenius norm
    """
    for get_task in [get_convnet_task, get_fullyconnect_task]:
        train_loader, net, loss_function = get_task(subs=3000)

        ispace_m2_generator = ISpace_M2Gradients(model=net,
                                                 dataloader=train_loader,
                                                 loss_function=loss_function)
        MIspace = DenseMatrix(ispace_m2_generator)

        m2 = M2Gradients(model=net,
                         dataloader=train_loader,
                         loss_function=loss_function)
        M = DenseMatrix(m2)

        n_examples = len(train_loader.sampler)
        check_ratio(M.trace(), MIspace.trace() / n_examples)

        MIspace_frob = MIspace.frobenius_norm()
        M_frob = M.frobenius_norm()
        check_ratio(M_frob, MIspace_frob / n_examples)


def test_pspace_implicit_vs_dense():
    """
    Check that the implicit representation and dense representation in
    Pspace actually compute the same values
    """
    for get_task in [get_convnet_task, get_fullyconnect_task]:
        train_loader, net, loss_function = get_task()

        m2_generator = M2Gradients(model=net,
                                   dataloader=train_loader,
                                   loss_function=loss_function)
        M_dense = DenseMatrix(m2_generator)
        M_implicit = ImplicitMatrix(m2_generator)

        dw = random_pvector(net)

        M_norm_imp = M_implicit.vTMv(dw)
        M_norm_den = M_dense.vTMv(dw)
        check_ratio(M_norm_den, M_norm_imp)

        trace_imp = M_implicit.trace()
        trace_den = M_dense.trace()
        check_ratio(trace_den, trace_imp)

        check_tensors(M_dense.mv(dw).get_flat_representation(),
                      M_implicit.mv(dw).get_flat_representation())


def test_pspace_lowrank_vs_dense():
    """
    Check lowrank representation results against DenseMatrix
    representation results
    """
    for get_task in [get_convnet_task, get_fullyconnect_task]:
        train_loader, net, loss_function = get_task(bs=100, subs=500)

        m2_generator = M2Gradients(model=net,
                                   dataloader=train_loader,
                                   loss_function=loss_function)
        M_dense = DenseMatrix(m2_generator)
        M_lowrank = LowRankMatrix(m2_generator)

        check_tensors(M_dense.get_matrix(), M_lowrank.get_matrix())

        dw = random_pvector(net)

        M_norm_lr = M_lowrank.vTMv(dw)
        M_norm_den = M_dense.vTMv(dw)
        check_ratio(M_norm_den, M_norm_lr)

        check_tensors(M_dense.mv(dw).get_flat_representation(),
                      M_lowrank.mv(dw).get_flat_representation())

        trace_lr = M_lowrank.trace()
        trace_den = M_dense.trace()
        check_ratio(trace_den, trace_lr)

        frob_lr = M_lowrank.frobenius_norm()
        frob_den = M_dense.frobenius_norm()
        check_ratio(frob_den, frob_lr)


def test_pspace_lowrank():
    for get_task in [get_convnet_task, get_fullyconnect_task]:
        train_loader, net, loss_function = get_task(bs=100, subs=500)
        m2_generator = M2Gradients(model=net,
                                   dataloader=train_loader,
                                   loss_function=loss_function)
        M = LowRankMatrix(m2_generator)

        M.compute_eigendecomposition()
        evals, evecs = M.get_eigendecomposition()
        check_tensors(M.get_matrix(),
                      torch.mm(torch.mm(evecs, torch.diag(evals)), evecs.t()))

        # TODO improve this
        # check_tensors(M.project_to_diag(M.get_matrix()).get_flat_representation(),
        #               torch.diag(M.evals))

        # check evecs:
        for i in range(evecs.size(1)):
            v = evecs[:, i]
            norm_v = torch.norm(v)
            v = PVector(net, vector_repr=v)
            if not (norm_v > 0.999 and norm_v < 1.001):
                # TODO improve this
                print(i, norm_v)
            check_tensors(M.mv(v).get_flat_representation(),
                          v.get_flat_representation() * evals[i])


def test_pspace_diag_vs_dense():
    """
    Check Diag representation against DenseMatrix representation
    """
    for get_task in [get_convnet_task, get_fullyconnect_task]:
        train_loader, net, loss_function = get_task(bs=100, subs=500)

        m2_generator = M2Gradients(model=net,
                                   dataloader=train_loader,
                                   loss_function=loss_function)
        M_dense = DenseMatrix(m2_generator)
        M_diag = DiagMatrix(m2_generator)

        check_tensors(torch.diag(torch.diag(M_dense.get_matrix())),
                      M_diag.get_matrix())

        trace_diag = M_diag.trace()
        trace_den = M_dense.trace()
        check_ratio(trace_den, trace_diag)

        dw = random_pvector(net)
        dw_flat = dw.get_flat_representation()
        check_tensors(M_diag.mv(dw).get_flat_representation(),
                      torch.mv(torch.diag(torch.diag(M_dense.get_matrix())),
                               dw_flat))

        frob_diag = M_diag.frobenius_norm()
        frob_dense = torch.norm(torch.diag(M_dense.get_matrix()))
        check_ratio(frob_dense, frob_diag)

        m_norm_diag = M_diag.vTMv(dw)
        m_norm_dense = torch.dot(dw_flat, torch.mv(M_diag.get_matrix(),
                                                   dw_flat))
        check_ratio(m_norm_dense, m_norm_diag)


def test_pspace_diag():
    """
    Check basic operations on diag matrices
    """
    train_loader1, net1, loss_function1 = \
        get_fullyconnect_task(bs=100, subs=500)
    train_loader2, net2, loss_function2 = \
        get_fullyconnect_task(bs=100, subs=500)

    m2_generator1 = M2Gradients(model=net1,
                                dataloader=train_loader1,
                                loss_function=loss_function1)
    m2_generator2 = M2Gradients(model=net2,
                                dataloader=train_loader2,
                                loss_function=loss_function2)
    M_diag1 = DiagMatrix(m2_generator1)
    M_diag2 = DiagMatrix(m2_generator2)

    sum_M = M_diag1 + M_diag2
    trace_1 = M_diag1.trace()
    trace_2 = M_diag2.trace()
    trace_sum = sum_M.trace()
    check_ratio(trace_1 + trace_2, trace_sum)

    check_tensors(M_diag1.get_matrix() + M_diag2.get_matrix(),
                  sum_M.get_matrix())

    sub_M = M_diag1 - M_diag2
    trace_sub = sub_M.trace()
    check_ratio(trace_1 - trace_2, trace_sub)

    check_tensors(M_diag1.get_matrix() - M_diag2.get_matrix(),
                  sub_M.get_matrix())


def test_pspace_dense():
    """
    Check basic operations on dense matrices
    """
    train_loader1, net1, loss_function1 = \
        get_fullyconnect_task(bs=100, subs=500)
    train_loader2, net2, loss_function2 = \
        get_fullyconnect_task(bs=100, subs=500)

    m2_generator1 = M2Gradients(model=net1,
                                dataloader=train_loader1,
                                loss_function=loss_function1)
    m2_generator2 = M2Gradients(model=net2,
                                dataloader=train_loader2,
                                loss_function=loss_function2)
    M_dense1 = DenseMatrix(m2_generator1)
    M_dense2 = DenseMatrix(m2_generator2)

    sum_M = M_dense1 + M_dense2
    trace_1 = M_dense1.trace()
    trace_2 = M_dense2.trace()
    trace_sum = sum_M.trace()
    check_ratio(trace_1 + trace_2, trace_sum)

    check_tensors(M_dense1.get_matrix() + M_dense2.get_matrix(),
                  sum_M.get_matrix())

    sub_M = M_dense1 - M_dense2
    trace_sub = sub_M.trace()
    check_ratio(trace_1 - trace_2, trace_sub)

    check_tensors(M_dense1.get_matrix() - M_dense2.get_matrix(),
                  sub_M.get_matrix())


def test_ispace_dense_vs_implicit():
    """
    Check implicit ISpace representation vs Dense
    """
    train_loader, net, loss_function = get_fullyconnect_task()

    ispace_m2 = ISpace_M2Gradients(model=net,
                                   dataloader=train_loader,
                                   loss_function=loss_function)
    M_dense = DenseMatrix(ispace_m2)
    M_implicit = ImplicitMatrix(ispace_m2)

    n_examples = len(train_loader.sampler)
    v = torch.rand((n_examples,), device='cuda')
    v = IVector(net, vector_repr=v)

    m_norm_dense = M_dense.vTMv(v)
    m_norm_implicit = M_implicit.vTMv(v)
    check_ratio(m_norm_dense, m_norm_implicit)

    frob_norm_dense = M_dense.frobenius_norm()
    frob_norm_implicit = M_implicit.frobenius_norm()
    check_ratio(frob_norm_dense, frob_norm_implicit)


def test_pspace_blockdiag_vs_dense():
    for get_task in [get_convnet_task, get_fullyconnect_task]:
        train_loader, net, loss_function = get_task()

        m2_generator = M2Gradients(model=net,
                                   dataloader=train_loader,
                                   loss_function=loss_function)
        M_dense = DenseMatrix(m2_generator)
        M_blockdiag = BlockDiagMatrix(m2_generator)

        eps = 1e-5
        G_dense = M_dense.get_matrix()
        G_blockdiag = M_blockdiag.get_matrix()
        mods, p_pos = get_individual_modules(net)
        for mod in mods:
            pos = p_pos[mod]
            numel = mod.weight.numel() + mod.bias.numel()
            # check that the blocks are equal
            check_tensors(G_dense[pos:pos+numel, pos:pos+numel],
                          G_blockdiag[pos:pos+numel, pos:pos+numel])
            # check that the rest is 0
            assert torch.norm(G_blockdiag[pos+numel:, pos:pos+numel]) < eps
            assert torch.norm(G_blockdiag[pos:pos+numel, pos+numel:]) < eps

        trace_bd = M_blockdiag.trace()
        trace_den = M_dense.trace()
        check_ratio(trace_den, trace_bd)

        random_v = random_pvector(net)
        random_v_flat = random_v.get_flat_representation()

        m_norm_blockdiag = M_blockdiag.vTMv(random_v)
        m_norm_direct = torch.dot(torch.mv(G_blockdiag, random_v_flat),
                                  random_v_flat)
        check_ratio(m_norm_direct, m_norm_blockdiag)

        frob_blockdiag = M_blockdiag.frobenius_norm()
        frob_frommatrix = torch.norm(G_blockdiag)
        check_ratio(frob_frommatrix, frob_blockdiag)

        check_tensors(torch.mv(M_blockdiag.get_matrix(), random_v_flat),
                      M_blockdiag.mv(random_v).get_flat_representation())
