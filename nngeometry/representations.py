import torch
from abc import ABC, abstractmethod
from .maths import kronecker

class AbstractMatrix(ABC):

    @abstractmethod
    def __init__(self, generator):
        return NotImplementedError

class DenseMatrix(AbstractMatrix):
    def __init__(self, generator, compute_eigendecomposition=False):
        self.generator = generator
        self.data = generator.get_matrix()
        if compute_eigendecomposition:
            self.compute_eigendecomposition()

    def compute_eigendecomposition(self, impl='symeig'):
        if impl == 'symeig':
            self.evals, self.evecs = torch.symeig(self.data, eigenvectors=True)
        elif impl == 'svd':
            _, self.evals, self.evecs = torch.svd(self.data, some=False)

    def mv(self, v):
        return torch.mv(self.data, v)

    def m_norm(self, v):
        return torch.dot(v, torch.mv(self.data, v)) ** .5

    def frobenius_norm(self):
        return torch.norm(self.data)

    def project_to_diag(self, v):
        if v.dim() == 1:
            return torch.mv(self.evecs.t(), v)
        elif v.dim() == 2:
            return torch.mm(torch.mm(self.evecs.t(), v), self.evecs)
        else:
            raise NotImplementedError

    def project_from_diag(self, v):
        return torch.mv(self.evecs, v)

    def get_eigendecomposition(self):
        return self.evals, self.evecs

    def size(self, *args):
        return self.data.size(*args)

    def trace(self):
        return self.data.trace()

    def get_matrix(self):
        return self.data

class DiagMatrix(AbstractMatrix):
    def __init__(self, generator):
        self.generator = generator
        self.data = generator.get_diag()

    def mv(self, v):
        return v * self.data

    def trace(self):
        return self.data.sum()

    def get_matrix(self):
        return torch.diag(self.data)

    def size(self, dim=None):
        s = self.data.size(0)
        if dim == 0 or dim == 1:
            return s
        elif dim is None:
            return (s, s)
        else:
            raise IndexError

class BlockDiagMatrix(AbstractMatrix):
    def __init__(self, generator):
        self.generator = generator
        self.data = generator.get_layer_blocks()

    def trace(self):
        return sum([torch.trace(b) for b in self.data])

    def get_matrix(self):
        s = self.generator.get_n_parameters()
        M = torch.zeros((s, s), device=self.generator.get_device())
        start = 0
        for b in self.data:
            M[start:start+b.size(0), start:start+b.size(0)].add_(b)
            start += b.size(0)
        return M

    def mv(self, vs):
        return [torch.mv(b, v) for b, v in zip(self.data, vs)]

class KFACMatrix(AbstractMatrix):
    def __init__(self, generator):
        self.generator = generator
        self.data = generator.get_kfac_blocks()

    def trace(self):
        return sum([torch.trace(a) * torch.trace(g) for a, g in self.data])

    def get_matrix(self, split_weight_bias=False):
        """
        - split_weight_bias (bool): if True then the parameters are ordered in
        the same way as in the dense or blockdiag representation, but it
        involves more operations. Otherwise the coefficients corresponding
        to the bias are mixed between coefficients of the weight matrix
        """
        s = self.generator.get_n_parameters()
        M = torch.zeros((s, s), device=self.generator.get_device())
        start = 0
        for a, g in self.data:
            sAG = a.size(0) * g.size(0)
            if split_weight_bias:
                reconstruct = torch.cat([torch.cat([kronecker(g, a[:-1,:-1]), kronecker(g, a[:-1,-1:])], dim=1),
                                         torch.cat([kronecker(g, a[-1:,:-1]), kronecker(g, a[-1:,-1:])], dim=1)], dim=0)
                M[start:start+sAG, start:start+sAG].add_(reconstruct)
            else:
                M[start:start+sAG, start:start+sAG].add_(kronecker(g, a))
            start += sAG
        return M

    # def mv(self, vs):
    #     return [torch.mv(b, v) for b, v in zip(self.data, vs)]

class ImplicitMatrix(AbstractMatrix):
    def __init__(self, generator):
        self.generator = generator

    def mv(self, v):
        return self.generator.implicit_mv(v)

    def m_norm(self, v):
        return self.generator.implicit_m_norm(v)

    def frobenius_norm(self):
        return self.generator.implicit_frobenius()

    def trace(self):
        return self.generator.implicit_trace()

    def size(self, dim=None):
        s = self.generator.get_n_parameters()
        if dim == 0 or dim == 1:
            return s
        elif dim is None:
            return (s, s)
        else:
            raise IndexError

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

    def mv(self, v):
        return torch.mv(self.data.t(), torch.mv(self.data, v))

    def compute_eigendecomposition(self, impl='symeig'):
        if impl == 'symeig':
            self.evals, V = torch.symeig(torch.mm(self.data, self.data.t()), eigenvectors=True)
            self.evecs = torch.mm(self.data.t(), V) / (self.evals**.5).unsqueeze(0)
        else:
            raise NotImplementedError

    def project_to_diag(self, v):
        if v.dim() == 1:
            return torch.mv(self.evecs.t(), v)
        elif v.dim() == 2:
            return torch.mm(torch.mm(self.evecs.t(), v), self.evecs)

    def get_eigendecomposition(self):
        return self.evals, self.evecs

class KrylovLowRankMatrix(AbstractMatrix):
    def __init__(self, generator):
        raise NotImplementedError()
