from abc import ABC, abstractmethod

import torch

from .vector import FVector, PVector


class PFMap(ABC):
    # a class for objects that link the parameter space
    # to the function space, i.e. PullBack or PushForward
    # or equivalently jacobian matrices
    pass


class PFMapDense(PFMap):
    def __init__(self, layer_collection, generator, data=None, examples=None):
        self.generator = generator
        self.layer_collection = layer_collection
        if data is not None:
            self.data = data
        else:
            self.data = generator.get_jacobian(examples, layer_collection)

    def to_torch(self):
        return self.data

    def size(self):
        return self.data.size()

    def jvp(self, v):
        v_flat = torch.mv(self.data.view(-1, self.data.size(-1)), v.to_torch())
        v_flat = v_flat.view(self.data.size(0), self.data.size(1))
        return FVector(vector_repr=v_flat)

    def vjp(self, v):
        v_flat = torch.mv(
            self.data.view(-1, self.data.size(-1)).t(),
            v.to_torch().view(-1),
        )
        return PVector(self.layer_collection, vector_repr=v_flat)

    def __add__(self, other):
        return PFMapDense(
            layer_collection=self.layer_collection,
            generator=self.generator,
            data=self.data + other.data,
        )

    def __sub__(self, other):
        return PFMapDense(
            layer_collection=self.layer_collection,
            generator=self.generator,
            data=self.data - other.data,
        )

    def __rmul__(self, x):
        return PFMapDense(
            layer_collection=self.layer_collection,
            generator=self.generator,
            data=x * self.data,
        )

    def iter_by_layer(self):
        layer_collection = self.layer_collection
        for layer_id, layer in layer_collection.layers.items():
            yield layer_id, layer, self.to_torch_layer(layer_id)

    def to_torch_layer(self, layer_id):
        start = self.layer_collection.p_pos[layer_id]
        layer = self.layer_collection.layers[layer_id]
        w = self.data[:, :, start : start + layer.weight.numel()].view(
            self.data.size(0), self.data.size(1), *layer.weight.size
        )
        start += layer.weight.numel()
        if layer.has_bias():
            b = self.data[:, :, start : start + layer.bias.numel()].view(
                self.data.size(0), self.data.size(1), *layer.bias.size
            )
            return (w, b)
        else:
            return (w,)

    def from_dict(layer_collection, generator, data_dict):
        parts = []
        for k in layer_collection.layers.keys():
            d = data_dict[k]
            parts.append(d[0].reshape(d[0].size(0), d[0].size(1), -1))
            if len(d) == 2:
                parts.append(d[1].reshape(d[0].size(0), d[0].size(1), -1))

        dense_jacobian = torch.cat(parts, dim=2)
        return PFMapDense(
            generator=generator, data=dense_jacobian, layer_collection=layer_collection
        )

    def to(self, **kwargs):
        return PFMapDense(
            layer_collection=self.layer_collection,
            generator=self.generator,
            data=self.data.to(**kwargs),
        )


class PFMapImplicit(PFMap):
    def __init__(self, layer_collection, generator, data=None, examples=None):
        self.layer_collection = layer_collection
        self.generator = generator
        self.examples = examples
        assert data is None

    def jvp(self, v):
        return self.generator.implicit_Jv(v, self.examples, self.layer_collection)
