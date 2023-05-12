import torch
from abc import ABC, abstractmethod
from .vector import FVector, PVector


class FMatAbstract(ABC):

    @abstractmethod
    def __init__(self, generator):
        return NotImplementedError


class FMatDense(FMatAbstract):
    def __init__(self, generator, data=None, examples=None):
        self.generator = generator
        if data is not None:
            self.data = data
        else:
            self.data = generator.get_gram_matrix(examples)

    def compute_eigendecomposition(self, impl='eigh'):
        s = self.data.size()
        M = self.data.view(s[0] * s[1], s[2] * s[3])
        if impl == 'eigh':
            self.evals, self.evecs = torch.linalg.eigh(M)
        elif impl == 'svd':
            _, self.evals, self.evecs = torch.svd(M, some=False)
        else:
            raise NotImplementedError

    def mv(self, v):
        # TODO: test
        v_flat = torch.mv(self.data, v.get_flat_representation())
        return FVector(vector_repr=v_flat)

    def vTMv(self, v):
        v_flat = v.get_flat_representation().view(-1)
        sd = self.data.size()
        return torch.dot(v_flat,
                         torch.mv(self.data.view(sd[0]*sd[1], sd[2]*sd[3]),
                                  v_flat))

    def frobenius_norm(self):
        return torch.norm(self.data)

    def project_to_diag(self, v):
        # TODO: test
        return PVector(model=v.model,
                       vector_repr=torch.mv(self.evecs.t(),
                                            v.get_flat_representation()))

    def project_from_diag(self, v):
        # TODO: test
        return PVector(model=v.model,
                       vector_repr=torch.mv(self.evecs,
                                            v.get_flat_representation()))

    def get_eigendecomposition(self):
        # TODO: test
        return self.evals, self.evecs

    def size(self, *args):
        # TODO: test
        return self.data.size(*args)

    def trace(self):
        # TODO: test
        return torch.trace(self.data)

    def get_dense_tensor(self):
        return self.data

    def __add__(self, other):
        # TODO: test
        sum_data = self.data + other.data
        return FMatDense(generator=self.generator,
                           data=sum_data)

    def __sub__(self, other):
        # TODO: test
        sub_data = self.data - other.data
        return FMatDense(generator=self.generator,
                           data=sub_data)
