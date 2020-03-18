from collections import OrderedDict


class LayerCollection:

    def __init__(self, layers=None):
        if layers is None:
            self.layers = []
        else:
            self.layers = layers

    def from_model(model):
        layers = []
        for layer, mod in model.named_modules():
            mod_class = mod.__class__.__name__
            if mod_class in ['Linear', 'Conv2d', 'BatchNorm1d', 'BatchNorm2d']:
                layers.append('%s.%s' % (layer, str(mod)))
        return LayerCollection(layers)

    def get_module_layer_maps(self, model):
        layer_to_module = OrderedDict()
        module_to_layer = OrderedDict()
        for layer, mod in model.named_modules():
            mod_class = mod.__class__.__name__
            layer_id = '%s.%s' % (layer, str(mod))
            if layer_id in self.layers:
                layer_to_module[layer_id] = mod
                module_to_layer[mod] = layer_id
            elif mod_class in ['Linear', 'Conv2d', 'BatchNorm1d',
                               'BatchNorm2d']:
                raise ValueError()
        return layer_to_module, module_to_layer
