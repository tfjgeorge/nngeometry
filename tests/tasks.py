import os

import torch
import torch.nn as nn
import torch.nn.functional as tF
from torch.nn.modules.conv import ConvTranspose2d
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from nngeometry.layercollection import LayerCollection
from nngeometry.layers import Affine1d, Cosine1d, WeightNorm1d, WeightNorm2d

default_datapath = "tmp"
if "SLURM_TMPDIR" in os.environ:
    default_datapath = os.path.join(os.environ["SLURM_TMPDIR"], "data")

if torch.cuda.is_available():
    device = "cuda"

    def to_device(tensor):
        return tensor.to(device)

    def to_device_model(model):
        model.to("cuda")

else:
    device = "cpu"

    # on cpu we need to use double as otherwise ill-conditioning in sums
    # causes numerical instability
    def to_device(tensor):
        return tensor.double()

    def to_device_model(model):
        model.double()


class FCNet(nn.Module):
    def __init__(self, out_size=10, normalization="none"):
        if normalization not in [
            "none",
            "batch_norm",
            "weight_norm",
            "cosine",
            "affine",
        ]:
            raise NotImplementedError
        super(FCNet, self).__init__()
        layers = []
        sizes = [18 * 18, 10, 10, out_size]
        for i, (s_in, s_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            if normalization == "weight_norm":
                layers.append(WeightNorm1d(s_in, s_out))
            elif normalization == "cosine":
                layers.append(Cosine1d(s_in, s_out))
            else:
                layers.append(nn.Linear(s_in, s_out, bias=(normalization == "none")))
            if normalization == "batch_norm":
                layers.append(nn.BatchNorm1d(s_out))
            elif normalization == "affine":
                layers.append(Affine1d(s_out, bias=(i % 2 == 0)))
            layers.append(nn.ReLU())
        # remove last nonlinearity:
        layers.pop()
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x[:, :, 5:-5, 5:-5].contiguous()
        x = x.view(x.size(0), -1)
        return self.net(x)


class FCNetSegmentation(nn.Module):
    def __init__(self, out_size=10):
        super(FCNetSegmentation, self).__init__()
        layers = []
        self.out_size = out_size
        sizes = [18 * 18, 10, 10, 4 * 4 * out_size]
        for s_in, s_out in zip(sizes[:-1], sizes[1:]):
            layers.append(nn.Linear(s_in, s_out))
            layers.append(nn.ReLU())
        # remove last nonlinearity:
        layers.pop()
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x[:, :, 5:-5, 5:-5].contiguous()
        x = x.view(x.size(0), -1)
        return self.net(x).view(-1, self.out_size, 4, 4)


class ConvNet(nn.Module):
    def __init__(self, normalization="none", binary=False):
        super(ConvNet, self).__init__()
        self.normalization = normalization
        if normalization == "weight_norm":
            self.wn1 = WeightNorm2d(1, 6, 3, 2)
        else:
            self.conv1 = nn.Conv2d(1, 6, 3, 2, bias=(normalization == "none"))
        if self.normalization == "batch_norm":
            self.bn1 = nn.BatchNorm2d(6)
        elif self.normalization == "group_norm":
            self.gn1 = nn.GroupNorm(2, 6)
        self.conv2 = nn.Conv2d(6, 5, 4, 1, bias=False)
        self.conv3 = nn.Conv2d(5, 7, 3, 1, 1)
        if self.normalization == "weight_norm":
            self.wn2 = WeightNorm1d(7, 4)
        else:
            self.fc1 = nn.Linear(7, 4)
        if self.normalization == "batch_norm":
            self.bn2 = nn.BatchNorm1d(4)
        if binary:
            self.fc2 = nn.Linear(4, 1)
        else:
            self.fc2 = nn.Linear(4, 3)

    def forward(self, x):
        if self.normalization == "batch_norm":
            x = tF.relu(self.bn1(self.conv1(x)))
        elif self.normalization == "group_norm":
            x = tF.relu(self.gn1(self.conv1(x)))
        elif self.normalization == "weight_norm":
            x = tF.relu(self.wn1(x))
        else:
            x = tF.relu(self.conv1(x))
        x = tF.avg_pool2d(x, 2, 2)
        x = tF.relu(self.conv2(x))
        x = tF.avg_pool2d(x, 2, 2)
        x = tF.relu(self.conv3(x), inplace=True)
        x = x.view(-1, 7)
        if self.normalization == "batch_norm":
            x = self.bn2(self.fc1(x))
        elif self.normalization == "weight_norm":
            x = self.wn2(x)
        else:
            x = self.fc1(x)

        x = self.fc2(tF.relu(x))
        return x


class SmallConvNet(nn.Module):
    def __init__(self, normalization="none", binary=False):
        super(SmallConvNet, self).__init__()

        if binary:
            n_output = 1
        else:
            n_output = 3
        self.normalization = normalization
        if normalization == "weight_norm":
            self.l1 = WeightNorm2d(1, 6, 3, 2)
            self.l2 = WeightNorm2d(6, n_output, 2, 3)
        elif normalization == "transpose":
            self.l1 = ConvTranspose2d(1, 6, (3, 2), 2)
            self.l2 = ConvTranspose2d(6, n_output, (2, 3), 3, bias=False)
        else:
            raise NotImplementedError

    def forward(self, x):
        x = x[:, :, 5:-5, 5:-5]
        x = tF.relu(self.l1(x))
        x = tF.relu(self.l2(x))
        return x.sum(dim=(2, 3))


class LinearFCNet(nn.Module):
    def __init__(self):
        super(LinearFCNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 3)
        self.fc2 = nn.Linear(28 * 28, 7, bias=False)

    def forward(self, x):
        fc1_out = self.fc1(x.view(x.size(0), -1))
        fc2_out = self.fc2(x.view(x.size(0), -1))
        output = torch.stack([fc1_out.sum(dim=(1)), fc2_out.sum(dim=(1))], dim=1)
        return output


def get_linear_fc_task():
    train_set = get_mnist(n_classes=2, subset=70)
    train_loader = DataLoader(dataset=train_set, batch_size=30, shuffle=False)
    net = LinearFCNet()
    to_device_model(net)
    net.eval()

    def output_fn(input, target):
        return net(to_device(input))

    layer_collection = LayerCollection.from_model(net)
    return (train_loader, layer_collection, net.parameters(), net, output_fn)


class LinearConvNet(nn.Module):
    def __init__(self):
        super(LinearConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, 3, 1)
        self.conv2 = nn.Conv2d(1, 3, 2, 1, bias=False)

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(x)
        output = torch.stack(
            [conv1_out.sum(dim=(1, 2, 3)), conv2_out.sum(dim=(1, 2, 3))], dim=1
        )
        return output


def get_linear_conv_task():
    train_set = get_mnist(n_classes=2, subset=70)
    train_loader = DataLoader(dataset=train_set, batch_size=30, shuffle=False)
    net = LinearConvNet()
    to_device_model(net)
    net.eval()

    def output_fn(input, target):
        return net(to_device(input))

    layer_collection = LayerCollection.from_model(net)
    return (train_loader, layer_collection, net.parameters(), net, output_fn)


class BatchNormFCLinearNet(nn.Module):
    def __init__(self):
        super(BatchNormFCLinearNet, self).__init__()
        self.fc0 = nn.Linear(28 * 28, 100)
        self.bn1 = nn.BatchNorm1d(100)
        self.bn2 = nn.BatchNorm1d(100)

    def forward(self, x):
        x = self.fc0(x.view(x.size(0), -1))
        bn1_out = self.bn1(x)
        bn2_out = self.bn2(-x)
        output = torch.stack([bn1_out.sum(dim=(1)), bn2_out.sum(dim=(1))], dim=1)
        return output


def get_batchnorm_fc_linear_task():
    train_set = get_mnist(subset=70)
    train_loader = DataLoader(dataset=train_set, batch_size=30, shuffle=False)
    net = BatchNormFCLinearNet()
    to_device_model(net)
    net.eval()

    def output_fn(input, target):
        return net(to_device(input))

    lc_full = LayerCollection.from_model(net)
    layer_collection = LayerCollection()
    # only keep fc1 and fc2
    layer_collection.add_layer(*lc_full.layers.popitem())
    layer_collection.add_layer(*lc_full.layers.popitem())
    parameters = list(net.bn2.parameters()) + list(net.bn1.parameters())

    return (train_loader, layer_collection, parameters, net, output_fn)


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
        output = torch.stack(
            [conv1_out.sum(dim=(1, 2, 3)), conv2_out.sum(dim=(1, 2, 3))], dim=1
        )
        return output


def get_batchnorm_conv_linear_task():
    train_set = get_mnist(subset=70)
    train_loader = DataLoader(dataset=train_set, batch_size=30, shuffle=False)
    net = BatchNormConvLinearNet()
    to_device_model(net)
    net.eval()

    def output_fn(input, target):
        return net(to_device(input))

    lc_full = LayerCollection.from_model(net)
    layer_collection = LayerCollection()
    # only keep fc1 and fc2
    layer_collection.add_layer(*lc_full.layers.popitem())
    layer_collection.add_layer(*lc_full.layers.popitem())
    parameters = list(net.conv2.parameters()) + list(net.conv1.parameters())

    return (train_loader, layer_collection, parameters, net, output_fn)


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
        self.bnfc = nn.BatchNorm1d(28 * 28)
        self.fc = nn.Linear(2352, 5)

    def forward(self, x):
        # artificially create a dataset with 2 channels by appending
        # transposed images as channel 2
        bs = x.size(0)
        two_channels = torch.cat([x, x.permute(0, 1, 3, 2)], dim=1)
        bnconv_out = self.bnconv(two_channels)
        bnfc_out = self.bnfc(x.view(bs, -1))
        stacked = torch.cat([bnconv_out.view(bs, -1), bnfc_out], dim=1)
        output = self.fc(stacked)
        return output


def get_batchnorm_nonlinear_task():
    train_set = get_mnist(subset=70)
    train_loader = DataLoader(dataset=train_set, batch_size=30, shuffle=False)
    net = BatchNormNonLinearNet()
    to_device_model(net)
    net.eval()

    def output_fn(input, target):
        return net(to_device(input))

    layer_collection = LayerCollection.from_model(net)
    return (train_loader, layer_collection, net.parameters(), net, output_fn)


def get_mnist(n_classes=None, subset=None):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    mnist = datasets.MNIST(
        root=default_datapath,
        train=True,
        download=True,
        transform=transform,
    )

    if subset is None:
        subset = len(mnist)
    loader = DataLoader(mnist, subset)
    x, y = next(iter(loader))

    if n_classes is not None:
        y = y % n_classes
    return TensorDataset(to_device(x), y.to(device))


def get_fullyconnect_task(normalization="none", binary=False):
    train_set = get_mnist(subset=70, n_classes=3)
    train_loader = DataLoader(dataset=train_set, batch_size=30, shuffle=False)
    if binary:
        net = FCNet(out_size=1, normalization=normalization)
    else:
        net = FCNet(out_size=3, normalization=normalization)
    to_device_model(net)
    net.eval()

    def output_fn(input, target):
        return net(to_device(input))

    layer_collection = LayerCollection.from_model(net)
    return (train_loader, layer_collection, net.parameters(), net, output_fn)


def get_fullyconnect_bn_task():
    return get_fullyconnect_task(normalization="batch_norm")


def get_fullyconnect_wn_task():
    return get_fullyconnect_task(normalization="weight_norm")


def get_fullyconnect_cosine_task():
    return get_fullyconnect_task(normalization="cosine")


def get_fullyconnect_affine_task():
    return get_fullyconnect_task(normalization="affine")


def get_conv_bn_task():
    return get_conv_task(normalization="batch_norm")


def get_conv_gn_task(binary=False):
    return get_conv_task(normalization="group_norm", binary=binary)


def get_conv_wn_task():
    return get_conv_task(normalization="weight_norm")


def get_conv_cosine_task():
    return get_conv_task(normalization="cosine")


def get_conv_task(normalization="none", small=False, binary=False):
    train_set = get_mnist(subset=70, n_classes=2 if binary else 3)
    train_loader = DataLoader(dataset=train_set, batch_size=30, shuffle=False)
    if small:
        net = SmallConvNet(normalization=normalization, binary=binary)
    else:
        net = ConvNet(normalization=normalization, binary=binary)
    to_device_model(net)
    net.eval()

    def output_fn(input, target):
        return net(to_device(input))

    layer_collection = LayerCollection.from_model(net)
    return (train_loader, layer_collection, net.parameters(), net, output_fn)


def get_small_conv_wn_task():
    return get_conv_task(normalization="weight_norm", small=True)


def get_small_conv_transpose_task():
    return get_conv_task(normalization="transpose", small=True)


def get_fullyconnect_onlylast_task():
    train_loader, lc_full, _, net, output_fn = get_fullyconnect_task()
    layer_collection = LayerCollection()
    # only keep last layer parameters
    layer_collection.add_layer(*lc_full.layers.popitem())
    parameters = net.net[-1].parameters()

    return train_loader, layer_collection, parameters, net, output_fn


def get_fullyconnect_segm_task():
    train_set = get_mnist(subset=70)
    train_loader = DataLoader(dataset=train_set, batch_size=30, shuffle=False)
    net = FCNetSegmentation(out_size=3)
    to_device_model(net)
    net.eval()

    def output_fn(input, target):
        return net(to_device(input))

    layer_collection = LayerCollection.from_model(net)
    return (train_loader, layer_collection, net.parameters(), net, output_fn)


class ConvNetWithSkipConnection(nn.Module):
    def __init__(self):
        super(ConvNetWithSkipConnection, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, 3, 1, 1)
        self.conv2 = nn.Conv2d(2, 2, 3, 1, 1)
        self.conv3 = nn.Conv2d(2, 2, 3, 1, 1)
        self.conv4 = nn.Conv2d(2, 3, 3, 1, 1)

    def forward(self, x):
        x_before_skip = tF.relu(self.conv1(x))
        x_block = tF.relu(self.conv2(x_before_skip))
        x_after_skip = tF.relu(self.conv3(x_block))
        x = tF.relu(self.conv4(x_after_skip + x_before_skip))
        x = x.sum(axis=(2, 3))
        return x


def get_conv_skip_task():
    train_set = get_mnist(subset=70)
    train_loader = DataLoader(dataset=train_set, batch_size=30, shuffle=False)
    net = ConvNetWithSkipConnection()
    to_device_model(net)
    net.eval()

    def output_fn(input, target):
        return net(to_device(input))

    layer_collection = LayerCollection.from_model(net)
    return (train_loader, layer_collection, net.parameters(), net, output_fn)


class Conv1dNet(nn.Module):
    def __init__(self, normalization="none"):
        super(Conv1dNet, self).__init__()
        if normalization != "none":
            raise NotImplementedError
        self.normalization = normalization
        self.conv1 = nn.Conv1d(1, 6, 3, 3)
        self.conv2 = nn.Conv1d(6, 5, 4, 8, bias=False)
        self.conv3 = nn.Conv1d(5, 2, 4, 4)
        self.fc1 = nn.Linear(16, 4)

    def forward(self, x):
        x = x.reshape(x.size(0), x.size(1), -1)
        x = tF.relu(self.conv1(x))
        x = tF.relu(self.conv2(x))
        x = tF.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


def get_conv1d_task(normalization="none"):
    train_set = get_mnist(subset=70)
    train_loader = DataLoader(dataset=train_set, batch_size=30, shuffle=False)
    net = Conv1dNet(normalization=normalization)
    to_device_model(net)
    net.eval()

    def output_fn(input, target):
        return net(to_device(input))

    layer_collection = LayerCollection.from_model(net)
    return (train_loader, layer_collection, net.parameters(), net, output_fn)


class LayerNormNet(nn.Module):
    def __init__(self, out_size):
        super(LayerNormNet, self).__init__()

        self.linear1 = nn.Linear(18 * 18, out_size)
        self.layer_norm1 = nn.LayerNorm((out_size,))

        self.net = nn.Sequential(self.linear1, self.layer_norm1)

    def forward(self, x):
        x = x[:, :, 5:-5, 5:-5].contiguous()
        x = x.view(x.size(0), -1)
        return self.net(x)


def get_layernorm_task():
    train_set = get_mnist(subset=70)
    train_loader = DataLoader(dataset=train_set, batch_size=30, shuffle=False)
    net = LayerNormNet(out_size=3)
    to_device_model(net)
    net.eval()

    def output_fn(input, target):
        return net(to_device(input))

    layer_collection = LayerCollection.from_model(net)
    return (train_loader, layer_collection, net.parameters(), net, output_fn)


class LayerNormConvNet(nn.Module):
    def __init__(self):
        super(LayerNormConvNet, self).__init__()
        self.layer = nn.Conv2d(1, 3, (3, 2), 2)
        self.layer_norm = nn.LayerNorm((3, 8, 9))

    def forward(self, x):
        x = x[:, :, 5:-5, 5:-5]
        x = self.layer(x)
        x = self.layer_norm(x)
        return x.sum(dim=(2, 3))


def get_layernorm_conv_task():
    train_set = get_mnist(subset=70)
    train_loader = DataLoader(dataset=train_set, batch_size=30, shuffle=False)
    net = LayerNormConvNet()
    to_device_model(net)
    net.eval()

    def output_fn(input, target):
        return net(to_device(input))

    layer_collection = LayerCollection.from_model(net)
    return (train_loader, layer_collection, net.parameters(), net, output_fn)
