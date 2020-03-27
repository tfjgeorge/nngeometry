import torch
import torch.nn as nn
import torch.nn.functional as tF
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from nngeometry.layercollection import LayerCollection
import os

default_datapath = 'tmp'
if 'SLURM_TMPDIR' in os.environ:
    default_datapath = os.path.join(os.environ['SLURM_TMPDIR'], 'data')


class FCNet(nn.Module):
    def __init__(self, in_size=10, out_size=10, n_hidden=2, hidden_size=15,
                 nonlinearity=nn.ReLU, batch_norm=False):
        super(FCNet, self).__init__()
        layers = []
        sizes = [in_size] + [hidden_size] * n_hidden + [out_size]
        for s_in, s_out in zip(sizes[:-1], sizes[1:]):
            layers.append(nn.Linear(s_in, s_out, bias=not batch_norm))
            if batch_norm:
                layers.append(nn.BatchNorm1d(s_out))
            layers.append(nonlinearity())
        # remove last nonlinearity:
        layers.pop()
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


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
        return self.fc1(x)


class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, 3, 1)
        self.conv2 = nn.Conv2d(1, 3, 2, 1, bias=False)
        self.fc1 = nn.Linear(28*28, 10)
        self.fc2 = nn.Linear(28*28, 7, bias=False)

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(x)
        fc1_out = self.fc1(x.view(x.size(0), -1))
        fc2_out = self.fc2(x.view(x.size(0), -1))
        output = torch.stack([conv1_out.sum(dim=(1, 2, 3)),
                              fc1_out.sum(dim=(1)),
                              conv2_out.sum(dim=(1, 2, 3)),
                              fc2_out.sum(dim=(1))], dim=1)
        return output


def get_linear_task():
    train_set = get_mnist()
    train_set = Subset(train_set, range(1000))
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=300,
        shuffle=False)
    net = LinearNet()
    net.to('cuda')

    def output_fn(input, target):
        return net(input.to('cuda'))

    layer_collection = LayerCollection.from_model(net)
    return (train_loader, layer_collection, net.parameters(),
            net, output_fn, 4)


class BatchNormLinearNet(nn.Module):
    def __init__(self):
        super(BatchNormLinearNet, self).__init__()
        self.conv1 = nn.BatchNorm2d(2)
        self.fc1 = nn.BatchNorm1d(28*28)

    def forward(self, x):
        # artificially create a dataset with 2 channels by appending
        # transposed images as channel 2
        two_channels = torch.cat([x, x.permute(0, 1, 3, 2)], dim=1)
        conv_out = self.conv1(two_channels)
        fc_out = self.fc1(x.view(x.size(0), -1))
        output = torch.stack([conv_out.sum(dim=(1, 2, 3)),
                              fc_out.sum(dim=(1))], dim=1)
        return output


def get_batchnorm_linear_task():
    train_set = get_mnist()
    train_set = Subset(train_set, range(1000))
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=1000,
        shuffle=False)
    net = BatchNormLinearNet()
    net.to('cuda')

    def output_fn(input, target):
        return net(input.to('cuda'))

    layer_collection = LayerCollection.from_model(net)
    return (train_loader, layer_collection, net.parameters(),
            net, output_fn, 2)


class BatchNormNonLinearNet(nn.Module):
    """
    BN Layer followed by a Linear Layer
    This is used to test jacobians against
    there linerization since this network does not
    suffer from the nonlinearity incurred by stacking
    a Linear Layer then a BN Layer
    """
    def __init__(self):
        super(BatchNormNonLinearNet, self).__init__()
        self.bnconv = nn.BatchNorm2d(2)
        self.bnfc = nn.BatchNorm1d(28*28)
        self.fc = nn.Linear(2352, 5)

    def forward(self, x):
        # artificially create a dataset with 2 channels by appending
        # transposed images as channel 2
        bs = x.size(0)
        two_channels = torch.cat([x, x.permute(0, 1, 3, 2)], dim=1)
        bnconv_out = self.bnconv(two_channels)
        bnfc_out = self.bnfc(x.view(bs, -1))
        stacked = torch.cat([bnconv_out.view(bs, -1),
                             bnfc_out], dim=1)
        output = self.fc(stacked)
        return output


def get_batchnorm_nonlinear_task():
    train_set = get_mnist()
    train_set = Subset(train_set, range(1000))
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=1000,
        shuffle=False)
    net = BatchNormNonLinearNet()
    net.to('cuda')

    def output_fn(input, target):
        return net(input.to('cuda'))

    layer_collection = LayerCollection.from_model(net)
    return (train_loader, layer_collection, net.parameters(),
            net, output_fn, 5)


def get_mnist():
    return datasets.MNIST(root=default_datapath,
                          train=True,
                          download=True,
                          transform=transforms.ToTensor())


def get_fullyconnect_task(batch_norm=False):
    train_set = get_mnist()
    train_set = Subset(train_set, range(1000))
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=300,
        shuffle=False)
    net = FCNet(in_size=784, batch_norm=batch_norm)
    net.to('cuda')

    def output_fn(input, target):
        return net(input.to('cuda'))

    layer_collection = LayerCollection.from_model(net)
    return (train_loader, layer_collection, net.parameters(),
            net, output_fn, 10)


def get_fullyconnect_bn_task():
    return get_fullyconnect_task(batch_norm=True)


def get_fullyconnect_onlylast_task():
    train_loader, lc_full, _, net, output_fn, n_output = \
        get_fullyconnect_task()
    layer_collection = LayerCollection()
    # only keep last layer parameters
    layer_collection.add_layer(*lc_full.layers.popitem())
    parameters = net.net[-1].parameters()

    return train_loader, layer_collection, parameters, net, output_fn, n_output
