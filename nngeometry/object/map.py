from abc import ABC, abstractmethod

import torch

from .vector import FVector, PVector


class PFMap(ABC):
    # a class for objects that link the parameter space
    # to the function space, i.e. PullBack or PushForward
    # or equivalently jacobian matrices
    pass


class PFMapDense(PFMap):
    def __init__(self, generator, data=None, examples=None):
        self.generator = generator
        if data is not None:
            self.data = data
        else:
            self.data = generator.get_jacobian(examples)

    def to_torch(self):
        return self.data

    def jvp(self, v):
        v_flat = torch.mv(self.data.view(-1, self.data.size(-1)), v.to_torch())
        v_flat = v_flat.view(self.data.size(0), self.data.size(1))
        return FVector(vector_repr=v_flat)

    def vjp(self, v):
        v_flat = torch.mv(
            self.data.view(-1, self.data.size(-1)).t(),
            v.to_torch().view(-1),
        )
        return PVector(self.generator.layer_collection, vector_repr=v_flat)

    def __add__(self, other):
        return PFMapDense(generator=self.generator, data=self.data + other.data)

    def __rmul__(self, x):
        return PFMapDense(generator=self.generator, data=x * self.data)

    def iter_by_module(self):
        layer_collection = self.generator.layer_collection
        for layer_id, layer in layer_collection.layers.items():
            start = layer_collection.p_pos[layer_id]
            w = self.data[:, :, start : start + layer.weight.numel()].view(
                self.data.size(0), self.data.size(1), *layer.weight.size
            )
            start += layer.weight.numel()
            if layer.bias is not None:
                b = self.data[:, :, start : start + layer.bias.numel()].view(
                    self.data.size(0), self.data.size(1), *layer.bias.size
                )
                start += layer.bias.numel()
                yield layer_id, (w, b)
            else:
                yield layer_id, (w,)

    def from_dict(generator, data_dict):
        parts = []
        for k, d in data_dict.items():
            parts.append(d[0].reshape(d[0].size(0), d[0].size(1), -1))
            if len(d) == 2:
                parts.append(d[1].reshape(d[0].size(0), d[0].size(1), -1))

        dense_jacobian = torch.cat(parts, dim=2)
        return PFMapDense(generator=generator, data=dense_jacobian)


class PFMapImplicit(PFMap):
    def __init__(self, generator, data=None, examples=None):
        self.generator = generator
        self.examples = examples
        assert data is None

    def jvp(self, v):
        return self.generator.implicit_Jv(v, self.examples)
