import torch
from .utils import get_individual_modules

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
            raise NotImplementedError
            # return self._flat_to_dict()

    def _dict_to_flat(self):
        parts = []
        for mod in get_individual_modules(self.model)[0]:
            parts.append(self.dict_repr[mod][0].view(-1))
            if len(self.dict_repr[mod]) > 1:
                parts.append(self.dict_repr[mod][1].view(-1))
        return torch.cat(parts)
