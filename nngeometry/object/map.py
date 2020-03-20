import torch
from abc import ABC, abstractmethod
from .vector import FVector, PVector


class AbstractPushForward(ABC):

    @abstractmethod
    def __init__(self, generator):
        return NotImplementedError


class DensePushForward(AbstractPushForward):
    def __init__(self, generator, data=None):
        self.generator = generator
        if data is not None:
            self.data = data
        else:
            self.data = generator.get_jacobian()

    def get_matrix(self):
        return self.data

    def mv(self, v):
        v_flat = torch.mv(self.data.view(-1, self.data.size(-1)),
                          v.get_flat_representation())
        v_flat = v_flat.view(self.data.size(0), self.data.size(1))
        return FVector(vector_repr=v_flat)


class ImplicitPushForward(AbstractPushForward):
    def __init__(self, generator):
        self.generator = generator

    def mv(self, v):
        return self.generator.implicit_Jv(v.get_flat_representation())


class AbstractPullBack(ABC):

    @abstractmethod
    def __init__(self, generator):
        return NotImplementedError


class DensePullBack(AbstractPullBack):
    def __init__(self, generator, data=None):
        self.generator = generator
        if data is not None:
            self.data = data
        else:
            self.data = generator.get_jacobian()

    def get_matrix(self):
        return self.data.t()

    def mv(self, v):
        v_flat = torch.mv(self.data.view(-1, self.data.size(-1)).t(),
                          v.get_flat_representation().view(-1))
        return PVector(self.generator.layer_collection, vector_repr=v_flat)
