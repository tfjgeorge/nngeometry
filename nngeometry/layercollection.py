from abc import ABC
from collections import OrderedDict


class LayerCollection:

    def __init__(self, layers=None):
        if layers is None:
            self.layers = OrderedDict()
            self._numel = 0
            self.p_pos = dict()
        else:
            self.layers = layers
            raise NotImplementedError

    def from_model(model):
        lc = LayerCollection()
        for layer, mod in model.named_modules():
            mod_class = mod.__class__.__name__
            if mod_class == 'Linear':
                lc.add_layer('%s.%s' % (layer, str(mod)),
                             LinearLayer(in_features=mod.in_features,
                                         out_features=mod.out_features,
                                         bias=(mod.bias is not None)))
            elif mod_class == 'Conv2d':
                lc.add_layer('%s.%s' % (layer, str(mod)),
                             Conv2dLayer(in_channels=mod.in_channels,
                                         out_channels=mod.out_channels,
                                         kernel_size=mod.kernel_size,
                                         bias=(mod.bias is not None)))
            elif mod_class == 'BatchNorm1d':
                lc.add_layer('%s.%s' % (layer, str(mod)),
                             BatchNorm1dLayer(num_features=mod.num_features))
            elif mod_class == 'BatchNorm2d':
                lc.add_layer('%s.%s' % (layer, str(mod)),
                             BatchNorm2dLayer(num_features=mod.num_features))

        return lc

    def get_layerid_module_maps(self, model):
        layerid_to_module = OrderedDict()
        module_to_layerid = OrderedDict()
        named_modules = {'%s.%s' % (l, str(m)): m
                         for l, m in model.named_modules()}
        for layer in self.layers.keys():
            layerid_to_module[layer] = named_modules[layer]
            module_to_layerid[named_modules[layer]] = layer
        return layerid_to_module, module_to_layerid

    def add_layer(self, name, layer):
        self.layers[name] = layer
        self.p_pos[name] = self._numel
        self._numel += layer.numel()

    def numel(self):
        return self._numel

    def __getitem__(self, layer_id):
        return self.layers[layer_id]

    def parameters(self, layerid_to_module):
        for layer_id, layer in self.layers.items():
            yield layerid_to_module[layer_id].weight
            if (isinstance(layer, BatchNorm1dLayer) or
                    isinstance(layer, BatchNorm2dLayer)):
                yield layerid_to_module[layer_id].bias
            # otherwise it is a Linear or Conv2d with optional bias
            elif layer.bias:
                yield layerid_to_module[layer_id].bias


class AbstractLayer(ABC):
    pass


class Conv2dLayer(AbstractLayer):

    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.bias = bias

    def numel(self):
        if self.bias:
            numel_bias = self.out_channels
        else:
            numel_bias = 0

        return (self.in_channels * self.out_channels * self.kernel_size[0] *
                self.kernel_size[1]) + numel_bias


class LinearLayer(AbstractLayer):

    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

    def numel(self):
        if self.bias:
            numel_bias = self.out_features
        else:
            numel_bias = 0

        return self.in_features * self.out_features + numel_bias


class BatchNorm1dLayer(AbstractLayer):

    def __init__(self, num_features):
        self.num_features = num_features

    def numel(self):
        return 2 * self.num_features


class BatchNorm2dLayer(AbstractLayer):

    def __init__(self, num_features):
        self.num_features = num_features

    def numel(self):
        return 2 * self.num_features
