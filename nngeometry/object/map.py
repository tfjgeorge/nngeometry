import torch
from abc import ABC, abstractmethod
from .vector import FVector


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
        v_flat = torch.mv(self.data, v.get_flat_representation())
        return FVector(v.model, vector_repr=v_flat)


class ImplicitPushForward(AbstractPushForward):
    def __init__(self, generator):
        self.generator = generator

    def mv(self, v):
        return self.generator.implicit_Jv(v.get_flat_representation())
