import torch
from ..utils import get_individual_modules
from ..layercollection import LayerCollection


def random_pvector_dict(layer_collection, device=None):
    v_dict = dict()
    for layer_id, layer in layer_collection.layers.items():
        if layer.bias is not None:
            v_dict[layer_id] = (torch.rand(layer.weight.size, device=device),
                                torch.rand(layer.bias.size, device=device))
        else:
            v_dict[layer_id] = (torch.rand(layer.weight.size))
    return PVector(layer_collection, dict_repr=v_dict)


def random_pvector(layer_collection, device=None):
    n_parameters = layer_collection.numel()
    random_v_flat = torch.rand((n_parameters,),
                               device=device) - .5
    random_v_flat /= torch.norm(random_v_flat)
    return PVector(layer_collection=layer_collection,
                   vector_repr=random_v_flat)


def random_fvector(n_samples, n_output=1, device=None):
    random_v_flat = torch.randn((n_output, n_samples,),
                                device=device)
    random_v_flat /= torch.norm(random_v_flat)
    return FVector(vector_repr=random_v_flat)


class PVector:
    """
    A vector in parameter space
    """
    def __init__(self, layer_collection, vector_repr=None,
                 dict_repr=None):
        self.layer_collection = layer_collection
        self.vector_repr = vector_repr
        self.dict_repr = dict_repr

    @staticmethod
    def from_model(model):
        dict_repr = dict()
        layer_collection = LayerCollection.from_model(model)
        l_to_m, _ = layer_collection.get_layerid_module_maps(model)
        for layer_id, layer in layer_collection.layers.items():
            mod = l_to_m[layer_id]
            if layer.bias is not None:
                dict_repr[layer_id] = (mod.weight, mod.bias)
            else:
                dict_repr[layer_id] = (mod.weight)
        return PVector(layer_collection, dict_repr=dict_repr)

    @staticmethod
    # TODO: fix and test
    def from_model_grad(model):
        dict_repr = dict()
        for mod in get_individual_modules(model)[0]:
            if mod.bias is not None:
                dict_repr[mod] = (mod.weight.grad, mod.bias.grad)
            else:
                dict_repr[mod] = (mod.weight.grad)
        return PVector(model, dict_repr=dict_repr)

    def clone(self):
        if self.dict_repr is not None:
            dict_clone = dict()
            for k, v in self.dict_repr.items():
                if len(v) == 2:
                    dict_clone[k] = (v[0].clone(), v[1].clone())
                else:
                    dict_clone[k] = (v[0].clone(),)
            return PVector(self.layer_collection, dict_repr=dict_clone)
        if self.vector_repr is not None:
            return PVector(self.layer_collection,
                           vector_repr=self.vector_repr.clone())

    def detach(self):
        if self.dict_repr is not None:
            dict_detach = dict()
            for k, v in self.dict_repr.items():
                if len(v) == 2:
                    dict_detach[k] = (v[0].detach(), v[1].detach())
                else:
                    dict_detach[k] = (v[0].detach(),)
            return PVector(self.layer_collection, dict_repr=dict_detach)
        if self.vector_repr is not None:
            return PVector(self.layer_collection,
                           vector_repr=self.vector_repr.detach())

    def get_flat_representation(self):
        if self.vector_repr is not None:
            return self.vector_repr
        elif self.dict_repr is not None:
            return self._dict_to_flat()
        else:
            return NotImplementedError

    def get_dict_representation(self):
        if self.dict_repr is not None:
            return self.dict_repr
        elif self.vector_repr is not None:
            return self._flat_to_dict()
        else:
            return NotImplementedError

    def _dict_to_flat(self):
        parts = []
        for layer_id, layer in self.layer_collection.layers.items():
            parts.append(self.dict_repr[layer_id][0].view(-1))
            if len(self.dict_repr[layer_id]) > 1:
                parts.append(self.dict_repr[layer_id][1].view(-1))
        return torch.cat(parts)

    def _flat_to_dict(self):
        dict_repr = dict()
        for layer_id, layer in self.layer_collection.layers.items():
            start = self.layer_collection.p_pos[layer_id]
            w = self.vector_repr[start:start+layer.weight.numel()] \
                    .view(*layer.weight.size)
            start += layer.weight.numel()
            if layer.bias is not None:
                b = self.vector_repr[start:start+layer.bias.numel()] \
                        .view(*layer.bias.size)
                start += layer.bias.numel()
                dict_repr[layer_id] = (w, b)
            else:
                dict_repr[layer_id] = (w,)
        return dict_repr

    def __rmul__(self, x):
        # TODO: test
        # scalar multiplication
        if self.dict_repr is not None:
            v_dict = dict()
            for l_id, l in self.layer_collection.layers.items():
                if l.bias:
                    v_dict[l_id] = (x * self.dict_repr[l_id][0],
                                    x * self.dict_repr[l_id][1])
                else:
                    v_dict[l_id] = (x * self.dict_repr[l_id][0])
            return PVector(self.layer_collection, dict_repr=v_dict)
        else:
            return PVector(self.layer_collection,
                           vector_repr=x * self.vector_repr)

    def __add__(self, other):
        if self.dict_repr is not None and other.dict_repr is not None:
            v_dict = dict()
            for l_id, l in self.layer_collection.layers.items():
                if l.bias is not None:
                    v_dict[l_id] = (self.dict_repr[l_id][0] +
                                    other.dict_repr[l_id][0],
                                    self.dict_repr[l_id][1] +
                                    other.dict_repr[l_id][1])
                else:
                    v_dict[l_id] = (self.dict_repr[l_id][0] +
                                    other.dict_repr[l_id][0])
            return PVector(self.layer_collection, dict_repr=v_dict)
        elif self.vector_repr is not None and other.vector_repr is not None:
            return PVector(self.layer_collection,
                           vector_repr=self.vector_repr+other.vector_repr)
        else:
            return PVector(self.layer_collection,
                           vector_repr=(self.get_flat_representation() +
                                        other.get_flat_representation()))

    def __sub__(self, other):
        if self.dict_repr is not None and other.dict_repr is not None:
            v_dict = dict()
            for l_id, l in self.layer_collection.layers.items():
                if l.bias is not None:
                    v_dict[l_id] = (self.dict_repr[l_id][0] -
                                    other.dict_repr[l_id][0],
                                    self.dict_repr[l_id][1] -
                                    other.dict_repr[l_id][1])
                else:
                    v_dict[l_id] = (self.dict_repr[l_id][0] -
                                    other.dict_repr[l_id][0])
            return PVector(self.layer_collection, dict_repr=v_dict)
        elif self.vector_repr is not None and other.vector_repr is not None:
            return PVector(self.layer_collection,
                           vector_repr=self.vector_repr-other.vector_repr)
        else:
            return PVector(self.layer_collection,
                           vector_repr=(self.get_flat_representation() -
                                        other.get_flat_representation()))


class FVector:
    """
    A vector in function space
    """
    def __init__(self, vector_repr=None):
        self.vector_repr = vector_repr

    def get_flat_representation(self):
        if self.vector_repr is not None:
            return self.vector_repr
        else:
            return NotImplementedError
