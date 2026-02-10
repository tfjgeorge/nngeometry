import warnings
from abc import ABC, abstractmethod

import torch

from .map import PFMap, PFMapDense
from .vector import FVector, PVector


class FMatAbstract(ABC):
    @abstractmethod
    def __init__(self, generator):
        return NotImplementedError

    def __matmul__(self, other):
        if isinstance(other, FVector):
            return self.mv(other)
        elif isinstance(other, PFMap):
            return self.mmap(other)
        else:
            return NotImplemented

    def __rmatmul__(self, other):
        if isinstance(other, FVector):
            return self.T.mv(other)
        elif isinstance(other, PFMap):
            return self.T.mmap(other)
        else:
            return NotImplemented


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
        v_flat = v.to_torch().view(-1)
        s = self.data.size()
        v_flat = torch.mv(self.data.view(s[0] * s[1], s[2] * s[3]), v_flat)
        return FVector(vector_repr=v_flat)

    def mmap(self, pfmap):
        s = self.data.size()
        pfmap_flat = pfmap.to_torch().view(s[2] * s[3], -1)
        return PFMapDense(
            self.layer_collection,
            self.generator,
            data=torch.mm(self.data.view(s[0] * s[1], s[2] * s[3]), pfmap_flat).view(
                s[0], s[1], -1
            ),
        )

    @property
    def T(self):
        return FMatDense(
            layer_collection=self.layer_collection,
            generator=self.generator,
            data=self.data.permute(2, 3, 0, 1),
        )

    def vTMv(self, v):
        v_flat = v.to_torch().view(-1)
        sd = self.data.size()
        return torch.dot(
            v_flat, torch.mv(self.data.view(sd[0] * sd[1], sd[2] * sd[3]), v_flat)
        )

    def frobenius_norm(self):
        warnings.warn(
            """Use norm(ord="fro") instead""", DeprecationWarning, stacklevel=2
        )
        return self.norm(ord="fro")

    def norm(self, ord=None):
        if ord is None or ord == "fro":
            return torch.sum(self.data**2) ** 0.5
        else:  # what should we do for 4D tensor ?
            raise RuntimeError(f"Order {ord} not supported.")

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
        sum_data = self.data + other.data
        return FMatDense(
            layer_collection=self.layer_collection,
            generator=self.generator,
            data=sum_data,
        )

    def __sub__(self, other):
        sub_data = self.data - other.data
        return FMatDense(
            layer_collection=self.layer_collection,
            generator=self.generator,
            data=sub_data,
        )

    def __rmul__(self, other):
        rmul_data = other * self.data
        return FMatDense(
            layer_collection=self.layer_collection,
            generator=self.generator,
            data=rmul_data,
        )

    def __imul__(self, other):
        self.data *= other
        return self
