from abc import ABC, abstractmethod

import torch

from .vector import FVector, PVector


class AbstractPushForward(ABC):
    @abstractmethod
    def __init__(self, generator):
        return NotImplementedError


class PushForwardDense(AbstractPushForward):
    def __init__(self, generator, data=None, examples=None):
        self.generator = generator
        if data is not None:
            self.data = data
        else:
            self.data = generator.get_jacobian(examples)

    def to_torch(self):
        return self.data

    def mv(self, v):
        v_flat = torch.mv(
            self.data.view(-1, self.data.size(-1)), v.to_torch()
        )
        v_flat = v_flat.view(self.data.size(0), self.data.size(1))
        return FVector(vector_repr=v_flat)


class PushForwardImplicit(AbstractPushForward):
    def __init__(self, generator, data=None, examples=None):
        self.generator = generator
        self.examples = examples
        assert data is None

    def mv(self, v):
        return self.generator.implicit_Jv(v, self.examples)


class PullBackAbstract(ABC):
    @abstractmethod
    def __init__(self, generator):
        return NotImplementedError


class PullBackDense(PullBackAbstract):
    def __init__(self, generator, data=None, examples=None):
        self.generator = generator
        if data is not None:
            self.data = data
        else:
            self.data = generator.get_jacobian(examples)

    def to_torch(self):
        return self.data

    def mv(self, v):
        v_flat = torch.mv(
            self.data.view(-1, self.data.size(-1)).t(),
            v.to_torch().view(-1),
        )
        return PVector(self.generator.layer_collection, vector_repr=v_flat)
