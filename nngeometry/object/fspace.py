from abc import ABC, abstractmethod

import torch

from .vector import FVector, PVector


class FMatAbstract(ABC):
    @abstractmethod
    def __init__(self, generator):
        return NotImplementedError


class FMatDense(FMatAbstract):
    def __init__(self, layer_collection, generator, data=None, examples=None):
        self.layer_collection = layer_collection
        self.generator = generator
        if data is not None:
            self.data = data
        else:
            self.data = generator.get_gram_matrix(examples, layer_collection)

    def compute_eigendecomposition(self, impl="eigh"):
        s = self.data.size()
        M = self.data.view(s[0] * s[1], s[2] * s[3])
        if impl == "eigh":
            self.evals, self.evecs = torch.linalg.eigh(M)
        elif impl == "svd":
            _, self.evals, self.evecs = torch.svd(M, some=False)
        else:
            raise NotImplementedError

    def mv(self, v):
        # TODO: test
        v_flat = torch.mv(self.data, v.to_torch())
        return FVector(vector_repr=v_flat)

    def vTMv(self, v):
        v_flat = v.to_torch().view(-1)
        sd = self.data.size()
        return torch.dot(
            v_flat, torch.mv(self.data.view(sd[0] * sd[1], sd[2] * sd[3]), v_flat)
        )

    def frobenius_norm(self):
        return torch.norm(self.data)

    def project_to_diag(self, v):
        # TODO: test
        return PVector(
            model=v.model,
            vector_repr=torch.mv(self.evecs.t(), v.to_torch()),
        )

    def project_from_diag(self, v):
        # TODO: test
        return PVector(model=v.model, vector_repr=torch.mv(self.evecs, v.to_torch()))

    def get_eigendecomposition(self):
        # TODO: test
        return self.evals, self.evecs

    def size(self, *args):
        # TODO: test
        return self.data.size(*args)

    def trace(self):
        # TODO: test
        return torch.trace(self.data)

    def to_torch(self):
        return self.data

    def __add__(self, other):
        # TODO: test
        sum_data = self.data + other.data
        return FMatDense(generator=self.generator, data=sum_data)

    def __sub__(self, other):
        # TODO: test
        sub_data = self.data - other.data
        return FMatDense(generator=self.generator, data=sub_data)
