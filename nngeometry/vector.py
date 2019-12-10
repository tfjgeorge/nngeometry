import torch
from .utils import get_individual_modules, get_n_parameters


def random_pvector_dict(model):
    v_dict = dict()
    for m in get_individual_modules(model)[0]:
        if m.bias is not None:
            v_dict[m] = (torch.rand_like(m.weight),
                         torch.rand_like(m.bias))
        else:
            v_dict[m] = (torch.rand_like(m.weight))
    return PVector(model=model, dict_repr=v_dict)


def random_pvector(model):
    n_parameters = get_n_parameters(model)
    random_v_flat = torch.rand((n_parameters,),
                               device=next(model.parameters()).device)
    return PVector(model=model,
                   vector_repr=random_v_flat)


class PVector:
    """
    A vector in parameter space
    """
    def __init__(self, model, vector_repr=None, dict_repr=None):
        self.model = model
        self.vector_repr = vector_repr
        self.dict_repr = dict_repr
        self.mods, self.p_pos = get_individual_modules(model)

    def from_model(model):
        dict_repr = dict()
        for mod in get_individual_modules(model)[0]:
            if mod.bias is not None:
                dict_repr[mod] = (mod.weight, mod.bias)
            else:
                dict_repr[mod] = (mod.weight)
        return PVector(model, dict_repr=dict_repr)

    @staticmethod
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
            return PVector(self.model, dict_repr=dict_clone)
        if self.vector_repr is not None:
            return PVector(self.model,  vector_repr=self.vector_repr.clone())

    def detach(self):
        if self.dict_repr is not None:
            dict_detach = dict()
            for k, v in self.dict_repr.items():
                if len(v) == 2:
                    dict_detach[k] = (v[0].detach(), v[1].detach())
                else:
                    dict_detach[k] = (v[0].detach(),)
            return PVector(self.model, dict_repr=dict_detach)
        if self.vector_repr is not None:
            return PVector(self.model,  vector_repr=self.vector_repr.detach())

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
            w = self.vector_repr[start:start+mod.weight.numel()] \
                    .view(*mod.weight.size())
            start += mod.weight.numel()
            if mod.bias is not None:
                b = self.vector_repr[start:start+mod.bias.numel()] \
                        .view(*mod.bias.size())
                start += mod.bias.numel()
                dict_repr[mod] = (w, b)
            else:
                dict_repr[mod] = (w,)
        return dict_repr

    def __add__(self, other):
        if self.dict_repr is not None and other.dict_repr is not None:
            v_dict = dict()
            for m in self.mods:
                if m.bias is not None:
                    v_dict[m] = (self.dict_repr[m][0] + other.dict_repr[m][0],
                                 self.dict_repr[m][1] + other.dict_repr[m][1])
                else:
                    v_dict[m] = (self.dict_repr[m][0] + other.dict_repr[m][0])
            return PVector(self.model, dict_repr=v_dict)
        elif self.vector_repr is not None and other.vector_repr is not None:
            return PVector(self.model,
                           vector_repr=self.vector_repr+other.vector_repr)
        else:
            return PVector(self.model,
                           vector_repr=(self.get_flat_representation() +
                                        other.get_flat_representation()))

    def __sub__(self, other):
        if self.dict_repr is not None and other.dict_repr is not None:
            v_dict = dict()
            for m in self.mods:
                if m.bias is not None:
                    v_dict[m] = (self.dict_repr[m][0] - other.dict_repr[m][0],
                                 self.dict_repr[m][1] - other.dict_repr[m][1])
                else:
                    v_dict[m] = (self.dict_repr[m][0] - other.dict_repr[m][0])
            return PVector(self.model, dict_repr=v_dict)
        elif self.vector_repr is not None and other.vector_repr is not None:
            return PVector(self.model,
                           vector_repr=self.vector_repr-other.vector_repr)
        else:
            return PVector(self.model,
                           vector_repr=(self.get_flat_representation() -
                                        other.get_flat_representation()))


class FVector:
    """
    A vector in function space
    """
    def __init__(self, model, vector_repr=None):
        self.model = model
        self.vector_repr = vector_repr

    def get_flat_representation(self):
        if self.vector_repr is not None:
            return self.vector_repr
        else:
            return NotImplementedError
