import torch.nn as nn
from .datasets import get_mnist
from .device import to_device_model,to_device
from torch.utils.data import DataLoader, Subset
from nngeometry.layercollection import LayerCollection

class LayerNormNet(nn.Module):
    def __init__(self, out_size):
        super(LayerNormNet, self).__init__()

        self.linear1 = nn.Linear(18*18, out_size)
        self.layer_norm1 = nn.LayerNorm((out_size,))

        self.net = nn.Sequential(self.linear1, self.layer_norm1)

    def forward(self, x):
        x = x[:, :, 5:-5, 5:-5].contiguous()
        x = x.view(x.size(0), -1)
        return self.net(x)

def get_layernorm_task(normalization="none"):
    train_set = get_mnist()
    train_set = Subset(train_set, range(70))
    train_loader = DataLoader(dataset=train_set, batch_size=30, shuffle=False)
    net = LayerNormNet(out_size=3)
    to_device_model(net)
    net.eval()

    def output_fn(input, target):
        return net(to_device(input))

    layer_collection = LayerCollection.from_model(net)
    return (train_loader, layer_collection, net.parameters(), net, output_fn, 3)
