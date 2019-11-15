import torch
from .utils import get_individual_modules

def from_model(model):
    dict_repr = dict()
    for mod in get_individual_modules(model)[0]:
        if mod.bias is not None:
            dict_repr[mod] = (mod.weight, mod.bias)
        else:
            dict_repr[mod] = (mod.weight)
    return dict_repr

class Vector:
    def __init__(self, model, vector_repr=None, dict_repr=None):
        self.model = model
        self.vector_repr = vector_repr
        self.dict_repr = dict_repr

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
        for mod in get_individual_modules(self.model)[0]:
            parts.append(self.dict_repr[mod][0].view(-1))
            if len(self.dict_repr[mod]) > 1:
                parts.append(self.dict_repr[mod][1].view(-1))
        return torch.cat(parts)

    def _flat_to_dict(self):
        start = 0
        dict_repr = dict()
        for mod in get_individual_modules(self.model)[0]:
            w = self.vector_repr[start:start+mod.weight.numel()].view(*mod.weight.size())
            start += mod.weight.numel()
            if mod.bias is not None:
                b = self.vector_repr[start:start+mod.bias.numel()].view(*mod.bias.size())
                start += mod.bias.numel()
                dict_repr[mod] = (w, b)
            else:
                dict_repr[mod] = (w,)
        return dict_repr
