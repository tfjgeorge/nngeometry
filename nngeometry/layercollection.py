import operator
from abc import ABC
from collections import OrderedDict
from functools import reduce


class LayerCollection:
    """
    This class describes a set or subset of layers, that can be used
    in order to instantiate :class:`nngeometry.object.PVector` or
    :class:`nngeometry.object.PSpaceDense` objects

    :param layers:
    """

    _known_modules = [
        "Linear",
        "Conv2d",
        "BatchNorm1d",
        "BatchNorm2d",
        "GroupNorm",
        "WeightNorm1d",
        "WeightNorm2d",
        "Cosine1d",
        "Affine1d",
        "ConvTranspose2d",
        "Conv1d",
        "LayerNorm",
        "Embedding",
    ]

    def __init__(self, layers=None):
        self._numel = 0
        self.p_pos = dict()
        if layers is None:
            self.layers = OrderedDict()
        else:
            self.layers = layers
            for layer_id, layer in layers.items():
                self.p_pos[layer_id] = self._numel
                self._numel += layer.numel()

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
            if mod_class in LayerCollection._known_modules:
                lc.add_layer(layer, LayerCollection._module_to_layer(mod))
            elif not ignore_unsupported_layers:
                if len(list(mod.children())) == 0 and len(list(mod.parameters())) > 0:
                    raise Exception("I do not know what to do with layer " + str(mod))

        return lc

    def get_layerid_module_map(self, model):
        layerid_to_module = OrderedDict()
        named_modules = dict(model.named_modules())
        for layer in self.layers.keys():
            layerid_to_module[layer] = named_modules[layer]
        return layerid_to_module

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
        if module.__class__.__name__ not in LayerCollection._known_modules:
            raise NotImplementedError
        for layer, mod in model.named_modules():
            if mod is module:
                self.add_layer(layer, LayerCollection._module_to_layer(mod))

    def _module_to_layer(mod):
        mod_class = mod.__class__.__name__
        if mod_class == "Linear":
            return LinearLayer(
                in_features=mod.in_features,
                out_features=mod.out_features,
                bias=(mod.bias is not None),
            )
        elif mod_class == "Conv2d":
            return Conv2dLayer(
                in_channels=mod.in_channels,
                out_channels=mod.out_channels,
                kernel_size=mod.kernel_size,
                bias=(mod.bias is not None),
            )
        elif mod_class == "ConvTranspose2d":
            return ConvTranspose2dLayer(
                in_channels=mod.in_channels,
                out_channels=mod.out_channels,
                kernel_size=mod.kernel_size,
                bias=(mod.bias is not None),
            )
        elif mod_class == "Conv1d":
            return Conv1dLayer(
                in_channels=mod.in_channels,
                out_channels=mod.out_channels,
                kernel_size=mod.kernel_size,
                bias=(mod.bias is not None),
            )
        elif mod_class == "BatchNorm1d":
            return BatchNorm1dLayer(num_features=mod.num_features)
        elif mod_class == "BatchNorm2d":
            return BatchNorm2dLayer(num_features=mod.num_features)
        elif mod_class == "GroupNorm":
            return GroupNormLayer(
                num_groups=mod.num_groups, num_channels=mod.num_channels
            )
        elif mod_class == "WeightNorm1d":
            return WeightNorm1dLayer(
                in_features=mod.in_features, out_features=mod.out_features
            )
        elif mod_class == "WeightNorm2d":
            return WeightNorm2dLayer(
                in_channels=mod.in_channels,
                out_channels=mod.out_channels,
                kernel_size=mod.kernel_size,
            )
        elif mod_class == "Cosine1d":
            return Cosine1dLayer(
                in_features=mod.in_features, out_features=mod.out_features
            )
        elif mod_class == "Affine1d":
            return Affine1dLayer(
                num_features=mod.num_features, bias=(mod.bias is not None)
            )
        elif mod_class == "LayerNorm":
            return LayerNormLayer(
                normalized_shape=mod.normalized_shape, bias=(mod.bias is not None)
            )
        elif mod_class == "Embedding":
            return EmbeddingLayer(
                embedding_dim=mod.embedding_dim, num_embeddings=mod.num_embeddings
            )

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
            if isinstance(layer, BatchNorm1dLayer) or isinstance(
                layer, BatchNorm2dLayer
            ):
                yield layerid_to_module[layer_id].bias
            # otherwise it is a Linear or Conv2d with optional bias
            elif layer.bias:
                yield layerid_to_module[layer_id].bias

    def named_parameters(self, layerid_to_module):
        for layer_id, layer in self.layers.items():
            yield layer_id + ".weight", layerid_to_module[layer_id].weight
            if isinstance(layer, BatchNorm1dLayer) or isinstance(
                layer, BatchNorm2dLayer
            ):
                yield layer_id + ".bias", layerid_to_module[layer_id].bias
            # otherwise it is a Linear or Conv2d with optional bias
            elif layer.bias:
                yield layer_id + ".bias", layerid_to_module[layer_id].bias

    def __eq__(self, other):
        for layer_id in set(self.layers.keys()).union(set(other.layers.keys())):
            if (
                layer_id not in other.layers.keys()
                or layer_id not in self.layers.keys()
                or self.layers[layer_id] != other.layers[layer_id]
            ):
                return False
        return True

    def get_common_layers(self, other):
        for layer_id, layer in self.layers.items():
            if layer_id in other.layers and layer == other.layers[layer_id]:
                yield layer_id, layer

    def merge(self, other):
        return LayerCollection(
            layers=OrderedDict([(lid, l) for lid, l in self.get_common_layers(other)])
        )


class AbstractLayer(ABC):
    transposed = False

    def __repr__(self):
        repr = f"{self.__class__}\n - weight = {self.weight}"
        if self.has_bias():
            return repr + f"\n - bias   = {self.bias}"
        else:
            return repr

    def has_bias(self):
        return hasattr(self, "bias") and self.bias is not None


class Conv2dLayer(AbstractLayer):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight = Parameter(
            out_channels, in_channels, kernel_size[0], kernel_size[1]
        )
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
        return (
            self.in_channels == other.in_channels
            and self.out_channels == other.out_channels
            and self.kernel_size == other.kernel_size
        )


class ConvTranspose2dLayer(AbstractLayer):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight = Parameter(
            out_channels, in_channels, kernel_size[0], kernel_size[1]
        )
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
        return (
            self.in_channels == other.in_channels
            and self.out_channels == other.out_channels
            and self.kernel_size == other.kernel_size
        )


class Conv1dLayer(AbstractLayer):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight = Parameter(out_channels, in_channels, kernel_size[0])
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
        return (
            self.in_channels == other.in_channels
            and self.out_channels == other.out_channels
            and self.kernel_size == other.kernel_size
        )


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
        return (
            self.in_features == other.in_features
            and self.out_features == other.out_features
        )


class EmbeddingLayer(AbstractLayer):
    transposed = True

    def __init__(self, num_embeddings, embedding_dim):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(num_embeddings, embedding_dim)

    def numel(self):
        return self.weight.numel()

    def __eq__(self, other):
        return (
            self.num_embeddings == other.num_embeddings
            and self.embedding_dim == other.embedding_dim
        )


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


class LayerNormLayer(AbstractLayer):
    def __init__(self, normalized_shape, bias=True):
        self.weight = Parameter(*normalized_shape)
        if bias:
            self.bias = Parameter(*normalized_shape)
        else:
            self.bias = None

    def numel(self):
        if self.bias is not None:
            return self.weight.numel() + self.bias.numel()
        else:
            return self.weight.numel()

    def __eq__(self, other):
        return self.weight == other.weight and self.bias == other.bias


class GroupNormLayer(AbstractLayer):
    def __init__(self, num_groups, num_channels):
        self.num_channels = num_channels
        self.weight = Parameter(num_channels)
        self.bias = Parameter(num_channels)

    def numel(self):
        return self.weight.numel() + self.bias.numel()

    def __eq__(self, other):
        return self.num_channels == other.num_channels


class WeightNorm1dLayer(AbstractLayer):
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(out_features, in_features)
        self.bias = None

    def numel(self):
        return self.weight.numel()

    def __eq__(self, other):
        return (
            self.in_features == other.in_features
            and self.out_features == other.out_features
        )


class WeightNorm2dLayer(AbstractLayer):
    def __init__(self, in_channels, out_channels, kernel_size):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight = Parameter(
            out_channels, in_channels, kernel_size[0], kernel_size[1]
        )
        self.bias = None

    def numel(self):
        return self.weight.numel()

    def __eq__(self, other):
        return (
            self.in_channels == other.in_channels
            and self.out_channels == other.out_channels
            and self.kernel_size == other.kernel_size
        )


class Cosine1dLayer(AbstractLayer):
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(out_features, in_features)
        self.bias = None

    def numel(self):
        return self.weight.numel()

    def __eq__(self, other):
        return (
            self.in_features == other.in_features
            and self.out_features == other.out_features
        )


class Affine1dLayer(AbstractLayer):
    def __init__(self, num_features, bias=True):
        self.num_features = num_features
        self.weight = Parameter(num_features)
        if bias:
            self.bias = Parameter(num_features)
        else:
            self.bias = None

    def numel(self):
        if self.bias is not None:
            return self.weight.numel() + self.bias.numel()
        else:
            return self.weight.numel()

    def __eq__(self, other):
        return self.num_features == other.num_features


class Parameter(object):
    def __init__(self, *size):
        self.size = size

    def numel(self):
        return reduce(operator.mul, self.size, 1)

    def __eq__(self, other):
        return self.size == other.size

    def __repr__(self):
        return f"Parameter with shape {self.size}"
