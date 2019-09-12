from nngeometry import EmpiricalL2Loss
from subsampled_mnist import get_dataset
import torch
import torch.nn as nn
import torch.nn.functional as tF
from torch.utils.data import DataLoader

class Net(nn.Module):
    def __init__(self, in_size=40, out_size=10, n_hidden=2, hidden_size=10,
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

def test_empiricall2loss():
    train_set = get_dataset('train')
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=250,
        shuffle=False)
    net = Net(in_size=10)
    net.to('cuda')

    el2 = EmpiricalL2Loss(model=net, dataloader=train_loader)
    loss_closure = lambda input, target: tF.nll_loss(net(input), target, reduction='sum')
    M = el2.get_matrix(loss_closure)

    print(M)

    # compare with || l(w+dw) - l(w) ||_F for randomly sampled dw
    loss_closure = lambda input, target: tF.nll_loss(net(input), target, reduction='none')
    l_0 = get_l_vector(train_loader, loss_closure)
    eps = 1e-3
    for i in range(20):
        dw = torch.rand((M.size(0),), device='cuda')
        dw *= eps / torch.norm(dw)
        update_model(net, dw)
        l_upd = get_l_vector(train_loader, loss_closure)
        update_model(net, -dw)
        
        print(torch.norm(l_upd - l_0)**2 / len(train_loader.sampler), torch.dot(torch.mv(M, dw), dw))


