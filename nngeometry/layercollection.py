from abc import ABC
from collections import OrderedDict
from functools import reduce
import operator


class LayerCollection:
    """
    This class describes a set or subset of layers, that can be used
    in order to instantiate :class:`nngeometry.object.PVector` or
    :class:`nngeometry.object.PSpaceDense` objects

    :param layers:
    """

    def __init__(self, layers=None):
        if layers is None:
            self.layers = OrderedDict()
            self._numel = 0
            self.p_pos = dict()
        else:
            self.layers = layers
            raise NotImplementedError

    def from_model(model, ignore_unsupported_layers=False):
        """
        Constructs a new LayerCollection object by using all parameters
        of the model passed as argument.

        :param model: The PyTorch model
        :type model: `nn.Module`
        :param ignore_unsupported_layers: If false, will raise an error
        when model contains layers that are not supported yet. If true, will
        silently ignore the layer
        :type ignore_unsupported_layers: bool
        """
        lc = LayerCollection()
        for layer, mod in model.named_modules():
            mod_class = mod.__class__.__name__
            if mod_class in ['Linear', 'Conv2d', 'BatchNorm1d',
                             'BatchNorm2d', 'GroupNorm']:
                lc.add_layer('%s.%s' % (layer, str(mod)),
                             LayerCollection._module_to_layer(mod))
            elif not ignore_unsupported_layers:
                if len(list(mod.children())) == 0 and len(list(mod.parameters())) > 0:
                    raise Exception('I do not know what to do with layer ' + str(mod))

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

    def add_layer_from_model(self, model, module):
        """
        Add a layer by specifying the module corresponding
        to this layer (e.g. torch.nn.Linear or torch.nn.BatchNorm1d)

        :param model: The model defining the neural network
        :param module: The layer to be added
        """
        if module.__class__.__name__ not in \
                ['Linear', 'Conv2d', 'BatchNorm1d',
                 'BatchNorm2d', 'GroupNorm']:
            raise NotImplementedError
        for layer, mod in model.named_modules():
            if mod is module:
                self.add_layer('%s.%s' % (layer, str(mod)),
                               LayerCollection._module_to_layer(mod))

    def _module_to_layer(mod):
        mod_class = mod.__class__.__name__
        if mod_class == 'Linear':
            return LinearLayer(in_features=mod.in_features,
                               out_features=mod.out_features,
                               bias=(mod.bias is not None))
        elif mod_class == 'Conv2d':
            return Conv2dLayer(in_channels=mod.in_channels,
                               out_channels=mod.out_channels,
                               kernel_size=mod.kernel_size,
                               bias=(mod.bias is not None))
        elif mod_class == 'BatchNorm1d':
            return BatchNorm1dLayer(num_features=mod.num_features)
        elif mod_class == 'BatchNorm2d':
            return BatchNorm2dLayer(num_features=mod.num_features)
        elif mod_class == 'GroupNorm':
            return GroupNormLayer(num_groups=mod.num_groups,
                                  num_channels=mod.num_channels)

    def numel(self):
        """
        Total number of scalar parameters in this LayerCollection object

        :return: number of scalar parameters
        :rtype: int 
        """
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

    def __eq__(self, other):
        for layer_id in set(self.layers.keys()).union(set(other.layers.keys())):
            if (layer_id not in other.layers.keys()
                    or layer_id not in self.layers.keys()
                    or self.layers[layer_id] != other.layers[layer_id]):
                return False
        return True


class AbstractLayer(ABC):
    pass


class Conv2dLayer(AbstractLayer):

    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight = Parameter(out_channels, in_channels, kernel_size[0],
                                kernel_size[1])
        if bias:
            self.bias = Parameter(out_channels)
        else:
            self.bias = None

    def numel(self):
        if self.bias is not None:
            return self.weight.numel() + self.bias.numel()
        else:
            return self.weight.numel()

    def __eq__(self, other):
        return (self.in_channels == other.in_channels and
                self.out_channels == other.out_channels and
                self.kernel_size == other.kernel_size)


class LinearLayer(AbstractLayer):

    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(out_features, in_features)
        if bias:
            self.bias = Parameter(out_features)
        else:
            self.bias = None

    def numel(self):
        if self.bias is not None:
            return self.weight.numel() + self.bias.numel()
        else:
            return self.weight.numel()

    def __eq__(self, other):
        return (self.in_features == other.in_features and
                self.out_features == other.out_features)


class BatchNorm1dLayer(AbstractLayer):

    def __init__(self, num_features):
        self.num_features = num_features
        self.weight = Parameter(num_features)
        self.bias = Parameter(num_features)

    def numel(self):
        return self.weight.numel() + self.bias.numel()

    def __eq__(self, other):
        return self.num_features == other.num_features


class BatchNorm2dLayer(AbstractLayer):

    def __init__(self, num_features):
        self.num_features = num_features
        self.weight = Parameter(num_features)
        self.bias = Parameter(num_features)

    def numel(self):
        return self.weight.numel() + self.bias.numel()

    def __eq__(self, other):
        return self.num_features == other.num_features


class GroupNormLayer(AbstractLayer):

    def __init__(self, num_groups, num_channels):
        self.num_channels = num_channels
        self.weight = Parameter(num_channels)
        self.bias = Parameter(num_channels)

    def numel(self):
        return self.weight.numel() + self.bias.numel()

    def __eq__(self, other):
        return self.num_channels == other.num_channels


class Parameter(object):

    def __init__(self, *size):
        self.size = size

    def numel(self):
        return reduce(operator.mul, self.size, 1)
