import torch
from abc import ABC, abstractmethod

class AbstractMatrix(ABC):

    @abstractmethod
    def __init__(self, generator):
        return NotImplemented


class DenseMatrix(AbstractMatrix):
    def __init__(self, generator, compute_eigendecomposition=False):
        self.generator = generator
        self.data = generator.get_matrix()
        if compute_eigendecomposition:
            self.compute_eigendecomposition()

    def compute_eigendecomposition(self):
        self.evals, self.evecs = self.get_eigendecomposition()

    def mv(self, v):
        return torch.mv(self.data, v)

    def m_norm(self, v):
        return torch.dot(v, torch.mv(self.data, v)) ** .5

    def project_to_diag(self, v):
        return torch.mv(self.evecs.t(), v)

    def project_from_diag(self, v):
        return torch.mv(self.evecs, v)

    def size(self, *args):
        return self.data.size(*args)

    def get_eigendecomposition(self):
        return torch.symeig(self.data, eigenvectors=True)

    def trace(self):
        return self.data.trace()

    def get_matrix(self):
        return self.data

class ImplicitMatrix(AbstractMatrix):
    def __init__(self, generator):
        self.generator = generator

    def mv(self, v):
        return self.generator.implicit_mv(v)

    def m_norm(self, v):
        return self.generator.implicit_m_norm(v)

class LowRankMatrix(AbstractMatrix):
    def __init__(self, generator):
        self.generator = generator
        self.data = generator.get_lowrank_matrix()

    def m_norm(self, v):
        return (torch.mv(self.data, v)**2).sum() ** .5

    def get_matrix(self):
        # you probably don't want to do that: you are
        # loosing the benefit of having a low rank representation
        # of your matrix but instead compute the potentially
        # much larger dense matrix
        return torch.mm(self.data.t(), self.data)

class KrylovLowRankMatrix(AbstractMatrix):
    def __init__(self, generator):
        raise NotImplementedError()