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

class FCNet(nn.Module):
    def __init__(self, out_size=10, normalization='none'):
        super(FCNet, self).__init__()
        layers = []
        sizes = [18*18, 10, 10, out_size]
        for s_in, s_out in zip(sizes[:-1], sizes[1:]):
            layers.append(nn.Linear(s_in, s_out, bias=(normalization == 'none')))
            if normalization == 'batch_norm':
                layers.append(nn.BatchNorm1d(s_out))
            elif normalization != 'none':
                raise NotImplementedError
            layers.append(nn.ReLU())
        # remove last nonlinearity:
        layers.pop()
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x[:, :, 5:-5, 5:-5].contiguous()
        x = x.view(x.size(0), -1)
        return self.net(x)


class ConvNet(nn.Module):
    def __init__(self, normalization='none'):
        super(ConvNet, self).__init__()
        self.normalization = normalization
        self.conv1 = nn.Conv2d(1, 6, 3, 2, bias=(normalization == 'none'))
        if self.normalization == 'batch_norm':
            self.bn1 = nn.BatchNorm2d(6)
        elif self.normalization == 'group_norm':
            self.gn = nn.GroupNorm(2, 6)
        self.conv2 = nn.Conv2d(6, 5, 4, 1)
        self.conv3 = nn.Conv2d(5, 7, 3, 1, 1)
        self.fc1 = nn.Linear(1*1*7, 3)

    def forward(self, x):
        if self.normalization == 'batch_norm':
            x = tF.relu(self.bn1(self.conv1(x)))
        elif self.normalization == 'group_norm':
            x = tF.relu(self.gn(self.conv1(x)))
        else:
            x = tF.relu(self.conv1(x))
        x = tF.max_pool2d(x, 2, 2)
        x = tF.relu(self.conv2(x))
        x = tF.max_pool2d(x, 2, 2)
        x = tF.relu(self.conv3(x))
        x = x.view(-1, 1*1*7)
        return self.fc1(x)


class LinearFCNet(nn.Module):
    def __init__(self):
        super(LinearFCNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 3)
        self.fc2 = nn.Linear(28*28, 7, bias=False)

    def forward(self, x):
        fc1_out = self.fc1(x.view(x.size(0), -1))
        fc2_out = self.fc2(x.view(x.size(0), -1))
        output = torch.stack([fc1_out.sum(dim=(1)),
                              fc2_out.sum(dim=(1))], dim=1)
        return output


def get_linear_fc_task():
    train_set = get_mnist()
    train_set = Subset(train_set, range(1000))
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=300,
        shuffle=False)
    net = LinearFCNet()
    net.to(device)

    def output_fn(input, target):
        return net(to_device(input))

    layer_collection = LayerCollection.from_model(net)
    return (train_loader, layer_collection, net.parameters(),
            net, output_fn, 2)


class LinearConvNet(nn.Module):
    def __init__(self):
        super(LinearConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, 3, 1)
        self.conv2 = nn.Conv2d(1, 3, 2, 1, bias=False)

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(x)
        output = torch.stack([conv1_out.sum(dim=(1, 2, 3)),
                              conv2_out.sum(dim=(1, 2, 3))], dim=1)
        return output


def get_linear_conv_task():
    train_set = get_mnist()
    train_set = Subset(train_set, range(1000))
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=300,
        shuffle=False)
    net = LinearConvNet()
    net.to(device)

    def output_fn(input, target):
        return net(to_device(input))

    layer_collection = LayerCollection.from_model(net)
    return (train_loader, layer_collection, net.parameters(),
            net, output_fn, 2)


class BatchNormFCLinearNet(nn.Module):
    def __init__(self):
        super(BatchNormFCLinearNet, self).__init__()
        self.fc0 = nn.Linear(28*28, 100)
        self.bn1 = nn.BatchNorm1d(100)
        self.bn2 = nn.BatchNorm1d(100)

    def forward(self, x):
        x = self.fc0(x.view(x.size(0), -1))
        bn1_out = self.bn1(x)
        bn2_out = self.bn2(-x)
        output = torch.stack([bn1_out.sum(dim=(1)),
                              bn2_out.sum(dim=(1))], dim=1)
        return output


def get_batchnorm_fc_linear_task():
    train_set = get_mnist()
    train_set = Subset(train_set, range(1000))
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=300,
        shuffle=False)
    net = BatchNormFCLinearNet()
    net.to(device)

    def output_fn(input, target):
        return net(to_device(input))

    lc_full = LayerCollection.from_model(net)
    layer_collection = LayerCollection()
    # only keep fc1 and fc2
    layer_collection.add_layer(*lc_full.layers.popitem())
    layer_collection.add_layer(*lc_full.layers.popitem())
    parameters = list(net.bn2.parameters()) + \
        list(net.bn1.parameters())

    return (train_loader, layer_collection, parameters,
            net, output_fn, 2)


class BatchNormConvLinearNet(nn.Module):
    def __init__(self):
        super(BatchNormConvLinearNet, self).__init__()
        self.conv0 = nn.Conv2d(1, 5, 3, 3)
        self.conv1 = nn.BatchNorm2d(5)
        self.conv2 = nn.BatchNorm2d(5)

    def forward(self, x):
        conv0_out = self.conv0(x)
        conv1_out = self.conv1(conv0_out)
        conv2_out = self.conv2(-conv0_out)
        output = torch.stack([conv1_out.sum(dim=(1, 2, 3)),
                              conv2_out.sum(dim=(1, 2, 3))], dim=1)
        return output


def get_batchnorm_conv_linear_task():
    train_set = get_mnist()
    train_set = Subset(train_set, range(1000))
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=300,
        shuffle=False)
    net = BatchNormConvLinearNet()
    net.to(device)

    def output_fn(input, target):
        return net(to_device(input))

    lc_full = LayerCollection.from_model(net)
    layer_collection = LayerCollection()
    # only keep fc1 and fc2
    layer_collection.add_layer(*lc_full.layers.popitem())
    layer_collection.add_layer(*lc_full.layers.popitem())
    parameters = list(net.conv2.parameters()) + \
        list(net.conv1.parameters())

    return (train_loader, layer_collection, parameters,
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
    net.to(device)

    def output_fn(input, target):
        return net(to_device(input))

    layer_collection = LayerCollection.from_model(net)
    return (train_loader, layer_collection, net.parameters(),
            net, output_fn, 5)


def get_mnist():
    return datasets.MNIST(root=default_datapath,
                          train=True,
                          download=True,
                          transform=transforms.ToTensor())


def get_fullyconnect_task(normalization='none'):
    train_set = get_mnist()
    train_set = Subset(train_set, range(1000))
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=300,
        shuffle=False)
    net = FCNet(out_size=3, normalization=normalization)
    net.to(device)

    def output_fn(input, target):
        return net(to_device(input))

    layer_collection = LayerCollection.from_model(net)
    return (train_loader, layer_collection, net.parameters(),
            net, output_fn, 3)


def get_fullyconnect_bn_task():
    return get_fullyconnect_task(normalization='batch_norm')


def get_conv_task(normalization='none'):
    train_set = get_mnist()
    train_set = Subset(train_set, range(1000))
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=300,
        shuffle=False)
    net = ConvNet(normalization=normalization)
    net.to(device)

    def output_fn(input, target):
        return net(to_device(input))

    layer_collection = LayerCollection.from_model(net)
    return (train_loader, layer_collection, net.parameters(),
            net, output_fn, 3)


def get_conv_bn_task():
    return get_conv_task(normalization='batch_norm')


def get_conv_gn_task():
    return get_conv_task(normalization='group_norm')


def get_fullyconnect_onlylast_task():
    train_loader, lc_full, _, net, output_fn, n_output = \
        get_fullyconnect_task()
    layer_collection = LayerCollection()
    # only keep last layer parameters
    layer_collection.add_layer(*lc_full.layers.popitem())
    parameters = net.net[-1].parameters()

    return train_loader, layer_collection, parameters, net, output_fn, n_output
