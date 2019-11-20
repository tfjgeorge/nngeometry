import torch
from abc import ABC, abstractmethod
from .maths import kronecker
from .utils import get_individual_modules
from .vector import Vector

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
        return torch.mv(self.data, v.get_flat_representation())

    def vTMv(self, v):
        v_flat = v.get_flat_representation()
        return torch.dot(v_flat, torch.mv(self.data, v_flat))

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
        return v.get_flat_representation() * self.data

    def trace(self):
        return self.data.sum()

    def vTMv(self, v):
        v_flat = v.get_flat_representation()
        return torch.dot(v_flat, self.data * v_flat)

    def frobenius_norm(self):
        return torch.norm(self.data)

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
        return sum([torch.trace(b) for b in self.data.values()])

    def get_matrix(self):
        s = self.generator.get_n_parameters()
        M = torch.zeros((s, s), device=self.generator.get_device())
        mods, p_pos = get_individual_modules(self.generator.model)
        for mod in mods:
            b = self.data[mod]
            start = p_pos[mod]
            M[start:start+b.size(0), start:start+b.size(0)].add_(b)
        return M

    def mv(self, vs):
        vs_dict = vs.get_dict_representation()
        out_dict = dict()
        for m in vs_dict.keys():
            v = vs_dict[m][0].view(-1)
            if m.bias is not None:
                v = torch.cat([v, vs_dict[m][1].view(-1)])
            mv = torch.mv(self.data[m], v)
            mv_tuple = (mv[:m.weight.numel()].view(*m.weight.size()),)
            if m.bias is not None:
                mv_tuple = (mv_tuple[0], mv[m.weight.numel():].view(*m.bias.size()),)
            out_dict[m] = mv_tuple
        return Vector(model=vs.model, dict_repr=out_dict)

    def frobenius_norm(self):
        return sum([torch.norm(b)**2 for b in self.data.values()])**.5

    def vTMv(self, vector):
        vector_dict = vector.get_dict_representation()
        norm2 = 0
        for mod in vector_dict.keys():
            v = vector_dict[mod][0].view(-1)
            if len(vector_dict[mod]) > 1:
                v = torch.cat([v, vector_dict[mod][1].view(-1)])
            norm2 += torch.dot(torch.mv(self.data[mod], v), v)
        return norm2


class KFACMatrix(AbstractMatrix):
    def __init__(self, generator):
        self.generator = generator
        self.data = generator.get_kfac_blocks()

    def trace(self):
        return sum([torch.trace(a) * torch.trace(g) for a, g in self.data.values()])

    def get_matrix(self, split_weight_bias=False):
        """
        - split_weight_bias (bool): if True then the parameters are ordered in
        the same way as in the dense or blockdiag representation, but it
        involves more operations. Otherwise the coefficients corresponding
        to the bias are mixed between coefficients of the weight matrix
        """
        s = self.generator.get_n_parameters()
        M = torch.zeros((s, s), device=self.generator.get_device())
        mods, p_pos = get_individual_modules(self.generator.model)
        for mod in mods:
            a, g = self.data[mod]
            start = p_pos[mod]
            sAG = a.size(0) * g.size(0)
            if split_weight_bias:
                reconstruct = torch.cat([torch.cat([kronecker(g, a[:-1,:-1]), kronecker(g, a[:-1,-1:])], dim=1),
                                         torch.cat([kronecker(g, a[-1:,:-1]), kronecker(g, a[-1:,-1:])], dim=1)], dim=0)
                M[start:start+sAG, start:start+sAG].add_(reconstruct)
            else:
                M[start:start+sAG, start:start+sAG].add_(kronecker(g, a))
        return M

    def mv(self, vs):
        vs_dict = vs.get_dict_representation()
        out_dict = dict()
        for m in vs_dict.keys():
            v = vs_dict[m][0].view(vs_dict[m][0].size(0), -1)
            if m.bias is not None:
                v = torch.cat([v, vs_dict[m][1].unsqueeze(1)], dim=1)
            a, g = self.data[m]
            mv = torch.mm(torch.mm(g, v), a)
            if m.bias is None:
                mv_tuple = (mv,)
            else:
                mv_tuple = (mv[:, :-1].contiguous(), mv[:, -1:].contiguous())
            out_dict[m] = mv_tuple
        return Vector(model=vs.model, dict_repr=out_dict)

    def vTMv(self, vector):
        vector_dict = vector.get_dict_representation()
        norm2 = 0
        for mod in vector_dict.keys():
            v = vector_dict[mod][0].view(vector_dict[mod][0].size(0), -1)
            if len(vector_dict[mod]) > 1:
                v = torch.cat([v, vector_dict[mod][1].unsqueeze(1)], dim=1)
            a, g = self.data[mod]
            norm2 += torch.dot(torch.mm(torch.mm(g, v), a).view(-1), v.view(-1))
        return norm2

    def frobenius_norm(self):
        return sum([torch.trace(torch.mm(a, a)) * torch.trace(torch.mm(g, g))
                    for a, g in self.data.values()])**.5

class ImplicitMatrix(AbstractMatrix):
    def __init__(self, generator):
        self.generator = generator

    def mv(self, v):
        return self.generator.implicit_mv(v.get_flat_representation())

    def vTMv(self, v):
        return self.generator.implicit_vTMv(v.get_flat_representation())

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

    def vTMv(self, v):
        Av = torch.mv(self.data, v.get_flat_representation())
        return torch.dot(Av, Av)

    def get_matrix(self):
        # you probably don't want to do that: you are
        # loosing the benefit of having a low rank representation
        # of your matrix but instead compute the potentially
        # much larger dense matrix
        return torch.mm(self.data.t(), self.data)

    def mv(self, v):
        return torch.mv(self.data.t(), torch.mv(self.data, v.get_flat_representation()))

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

    def trace(self):
        return torch.trace(torch.mm(self.data, self.data.t()))

    def frobenius_norm(self):
        A = torch.mm(self.data, self.data.t())
        return torch.trace(torch.mm(A, A))**.5

class KrylovLowRankMatrix(AbstractMatrix):
    def __init__(self, generator):
        raise NotImplementedError()
