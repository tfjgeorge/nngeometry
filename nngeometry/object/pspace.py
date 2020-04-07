import torch
from abc import ABC, abstractmethod
from ..maths import kronecker
from ..utils import get_individual_modules
from .vector import PVector


class PSpaceAbstract(ABC):

    @abstractmethod
    def __init__(self, generator):
        raise NotImplementedError

    @abstractmethod
    def get_dense_tensor(self):
        raise NotImplementedError

    @abstractmethod
    def trace(self):
        raise NotImplementedError

    @abstractmethod
    def frobenius_norm(self):
        raise NotImplementedError

    @abstractmethod
    def mv(self, v):
        raise NotImplementedError

    @abstractmethod
    def vTMv(self, v):
        raise NotImplementedError

    @abstractmethod
    def inverse(self, regul):
        raise NotImplementedError

    @abstractmethod
    def get_diag(self):
        raise NotImplementedError

    def size(self, dim=None):
        # TODO: test
        s = self.generator.layer_collection.numel()
        if dim == 0 or dim == 1:
            return s
        elif dim is None:
            return (s, s)
        else:
            raise IndexError


class PSpaceDense(PSpaceAbstract):
    def __init__(self, generator, data=None):
        self.generator = generator
        if data is not None:
            self.data = data
        else:
            self.data = generator.get_covariance_matrix()

    def compute_eigendecomposition(self, impl='symeig'):
        # TODO: test
        if impl == 'symeig':
            self.evals, self.evecs = torch.symeig(self.data, eigenvectors=True)
        elif impl == 'svd':
            _, self.evals, self.evecs = torch.svd(self.data, some=False)
        else:
            raise NotImplementedError

    def solve(self, v, regul=1e-8, impl='solve'):
        """
        solves v = Ax in x
        """
        # TODO: test
        if impl == 'solve':
            # TODO: reuse LU decomposition once it is computed
            inv_v, _ = torch.solve(v.get_flat_representation().view(-1, 1),
                                   self.data +
                                   regul * torch.eye(self.size(0),
                                                     device=self.data.device))
            return PVector(v.layer_collection, vector_repr=inv_v[:, 0])
        elif impl == 'eigendecomposition':
            v_eigenbasis = self.project_to_diag(v)
            inv_v_eigenbasis = v_eigenbasis / (self.evals + regul)
            return self.project_from_diag(inv_v_eigenbasis)
        else:
            raise NotImplementedError

    def inverse(self, regul=1e-8):
        inv_tensor = torch.inverse(self.data +
                                   regul * torch.eye(self.size(0),
                                                     device=self.data.device))
        return PSpaceDense(generator=self.generator,
                           data=inv_tensor)

    def mv(self, v):
        v_flat = torch.mv(self.data, v.get_flat_representation())
        return PVector(v.layer_collection, vector_repr=v_flat)

    def vTMv(self, v):
        v_flat = v.get_flat_representation()
        return torch.dot(v_flat, torch.mv(self.data, v_flat))

    def frobenius_norm(self):
        return torch.norm(self.data)

    def project_to_diag(self, v):
        # TODO: test
        return torch.mv(self.evecs.t(), v.get_flat_representation())

    def project_from_diag(self, v):
        # TODO: test
        return PVector(layer_collection=self.generator.layer_collection,
                       vector_repr=torch.mv(self.evecs, v))

    def get_eigendecomposition(self):
        # TODO: test
        return self.evals, self.evecs

    def size(self, *args):
        # TODO: test
        return self.data.size(*args)

    def trace(self):
        return torch.trace(self.data)

    def get_dense_tensor(self):
        return self.data

    def get_diag(self):
        return torch.diag(self.data)

    def __add__(self, other):
        sum_data = self.data + other.data
        return PSpaceDense(generator=self.generator,
                           data=sum_data)

    def __sub__(self, other):
        sub_data = self.data - other.data
        return PSpaceDense(generator=self.generator,
                           data=sub_data)

    def __rmul__(self, x):
        return PSpaceDense(generator=self.generator,
                           data=x * self.data)


class PSpaceDiag(PSpaceAbstract):
    def __init__(self, generator=None, data=None):
        self.generator = generator
        if data is not None:
            self.data = data
        else:
            self.data = generator.get_covariance_diag()

    def inverse(self, regul=1e-8):
        inv_tensor = 1. / (self.data + regul)
        return PSpaceDiag(generator=self.generator,
                          data=inv_tensor)

    def mv(self, v):
        v_flat = v.get_flat_representation() * self.data
        return PVector(v.layer_collection, vector_repr=v_flat)

    def trace(self):
        return self.data.sum()

    def vTMv(self, v):
        v_flat = v.get_flat_representation()
        return torch.dot(v_flat, self.data * v_flat)

    def frobenius_norm(self):
        return torch.norm(self.data)

    def get_dense_tensor(self):
        return torch.diag(self.data)

    def get_diag(self):
        return self.data

    def __add__(self, other):
        sum_diags = self.data + other.data
        return PSpaceDiag(generator=self.generator,
                          data=sum_diags)

    def __sub__(self, other):
        sub_diags = self.data - other.data
        return PSpaceDiag(generator=self.generator,
                          data=sub_diags)

    def __rmul__(self, x):
        return PSpaceDiag(generator=self.generator,
                          data=x * self.data)


class PSpaceBlockDiag(PSpaceAbstract):
    def __init__(self, generator, data=None):
        self.generator = generator
        if data is not None:
            self.data = data
        else:
            self.data = generator.get_covariance_layer_blocks()

    def trace(self):
        # TODO test
        return sum([torch.trace(b) for b in self.data.values()])

    def get_dense_tensor(self):
        s = self.generator.layer_collection.numel()
        M = torch.zeros((s, s), device=self.generator.get_device())
        for layer_id in self.generator.layer_collection.layers.keys():
            b = self.data[layer_id]
            start = self.generator.layer_collection.p_pos[layer_id]
            M[start:start+b.size(0), start:start+b.size(0)].add_(b)
        return M

    def get_diag(self):
        diag = []
        for layer_id in self.generator.layer_collection.layers.keys():
            b = self.data[layer_id]
            diag.append(torch.diag(b))
        return torch.cat(diag)

    def mv(self, vs):
        # TODO test
        vs_dict = vs.get_dict_representation()
        out_dict = dict()
        for layer_id, layer in self.generator.layer_collection.layers.items():
            v = vs_dict[layer_id][0].view(-1)
            if layer.bias is not None:
                v = torch.cat([v, vs_dict[layer_id][1].view(-1)])
            mv = torch.mv(self.data[layer_id], v)
            mv_tuple = (mv[:layer.weight.numel()].view(*layer.weight.size),)
            if layer.bias is not None:
                mv_tuple = (mv_tuple[0],
                            mv[layer.weight.numel():].view(*layer.bias.size),)
            out_dict[layer_id] = mv_tuple
        return PVector(layer_collection=vs.layer_collection,
                       dict_repr=out_dict)

    def inverse(self, regul=1e-8):
        inv_data = dict()
        for layer_id, layer in self.generator.layer_collection.layers.items():
            b = self.data[layer_id]
            inv_b = torch.inverse(b +
                                  regul *
                                  torch.eye(b.size(0), device=b.device))
            inv_data[layer_id] = inv_b
        return PSpaceBlockDiag(generator=self.generator,
                               data=inv_data)

    def frobenius_norm(self):
        # TODO test
        return sum([torch.norm(b)**2 for b in self.data.values()])**.5

    def vTMv(self, vector):
        # TODO test
        vector_dict = vector.get_dict_representation()
        norm2 = 0
        for mod in vector_dict.keys():
            v = vector_dict[mod][0].view(-1)
            if len(vector_dict[mod]) > 1:
                v = torch.cat([v, vector_dict[mod][1].view(-1)])
            norm2 += torch.dot(torch.mv(self.data[mod], v), v)
        return norm2

    def __add__(self, other):
        sum_data = {l_id: d + other.data[l_id]
                    for l_id, d in self.data.items()}
        return PSpaceBlockDiag(generator=self.generator,
                               data=sum_data)

    def __sub__(self, other):
        sum_data = {l_id: d - other.data[l_id]
                    for l_id, d in self.data.items()}
        return PSpaceBlockDiag(generator=self.generator,
                               data=sum_data)

    def __rmul__(self, x):
        sum_data = {l_id: x * d for l_id, d in self.data.items()}
        return PSpaceBlockDiag(generator=self.generator,
                               data=sum_data)


class PSpaceKFAC(PSpaceAbstract):
    def __init__(self, generator, data=None):
        self.generator = generator
        if data is None:
            self.data = generator.get_kfac_blocks()
        else:
            self.data = data

    def trace(self):
        return sum([torch.trace(a) * torch.trace(g)
                    for a, g in self.data.values()])

    def inverse(self, regul=1e-8, use_pi=True):
        inv_data = dict()
        for layer_id, layer in self.generator.layer_collection.layers.items():
            a, g = self.data[layer_id]
            if use_pi:
                pi = (torch.trace(a) / torch.trace(g) *
                      g.size(0) / a.size(0))**.5
            else:
                pi = 1
            inv_a = torch.inverse(a +
                                  pi * regul**.5 *
                                  torch.eye(a.size(0), device=a.device))
            inv_g = torch.inverse(g +
                                  regul**.5 / pi *
                                  torch.eye(g.size(0), device=g.device))
            inv_data[layer_id] = (inv_a, inv_g)
        return PSpaceKFAC(generator=self.generator,
                          data=inv_data)

    def get_dense_tensor(self, split_weight_bias=True):
        """
        - split_weight_bias (bool): if True then the parameters are ordered in
        the same way as in the dense or blockdiag representation, but it
        involves more operations. Otherwise the coefficients corresponding
        to the bias are mixed between coefficients of the weight matrix
        """
        s = self.generator.layer_collection.numel()
        M = torch.zeros((s, s), device=self.generator.get_device())
        for layer_id, layer in self.generator.layer_collection.layers.items():
            a, g = self.data[layer_id]
            start = self.generator.layer_collection.p_pos[layer_id]
            sAG = a.size(0) * g.size(0)
            if split_weight_bias:
                reconstruct = torch.cat([
                    torch.cat([kronecker(g, a[:-1, :-1]),
                               kronecker(g, a[:-1, -1:])], dim=1),
                    torch.cat([kronecker(g, a[-1:, :-1]),
                               kronecker(g, a[-1:, -1:])], dim=1)], dim=0)
                M[start:start+sAG, start:start+sAG].add_(reconstruct)
            else:
                M[start:start+sAG, start:start+sAG].add_(kronecker(g, a))
        return M

    def get_diag(self, split_weight_bias=True):
        """
        - split_weight_bias (bool): if True then the parameters are ordered in
        the same way as in the dense or blockdiag representation, but it
        involves more operations. Otherwise the coefficients corresponding
        to the bias are mixed between coefficients of the weight matrix
        """
        diags = []
        for layer_id, layer in self.generator.layer_collection.layers.items():
            a, g = self.data[layer_id]
            diag_of_block = (torch.diag(g).view(-1, 1) *
                             torch.diag(a).view(1, -1))
            if split_weight_bias:
                diags.append(diag_of_block[:, :-1].contiguous().view(-1))
                diags.append(diag_of_block[:, -1:].view(-1))
            else:
                diags.append(diag_of_block.view(-1))
        return torch.cat(diags)

    def mv(self, vs):
        vs_dict = vs.get_dict_representation()
        out_dict = dict()
        for layer_id, layer in self.generator.layer_collection.layers.items():
            v = vs_dict[layer_id][0].view(vs_dict[layer_id][0].size(0), -1)
            if layer.bias is not None:
                v = torch.cat([v, vs_dict[layer_id][1].unsqueeze(1)], dim=1)
            a, g = self.data[layer_id]
            mv = torch.mm(torch.mm(g, v), a)
            if layer.bias is None:
                mv_tuple = (mv,)
            else:
                mv_tuple = (mv[:, :-1].contiguous(), mv[:, -1].contiguous())
            out_dict[layer_id] = mv_tuple
        return PVector(layer_collection=vs.layer_collection,
                       dict_repr=out_dict)

    def vTMv(self, vector):
        vector_dict = vector.get_dict_representation()
        norm2 = 0
        for layer_id, layer in self.generator.layer_collection.layers.items():
            v = vector_dict[layer_id][0].view(vector_dict[layer_id][0].size(0),
                                              -1)
            if len(vector_dict[layer_id]) > 1:
                v = torch.cat([v, vector_dict[layer_id][1].unsqueeze(1)],
                              dim=1)
            a, g = self.data[layer_id]
            norm2 += torch.dot(torch.mm(torch.mm(g, v), a).view(-1),
                               v.view(-1))
        return norm2

    def frobenius_norm(self):
        return sum([torch.trace(torch.mm(a, a)) * torch.trace(torch.mm(g, g))
                    for a, g in self.data.values()])**.5

    def compute_eigendecomposition(self, impl='symeig'):
        self.evals = dict()
        self.evecs = dict()
        mods, p_pos = get_individual_modules(self.generator.model)
        if impl == 'symeig':
            for mod in mods:
                a, g = self.data[mod]
                evals_a, evecs_a = torch.symeig(a, eigenvectors=True)
                evals_g, evecs_g = torch.symeig(g, eigenvectors=True)
                self.evals[mod] = (evals_a, evals_g)
                self.evecs[mod] = (evecs_a, evecs_g)
        else:
            raise NotImplementedError

    def get_eigendecomposition(self):
        return self.evals, self.evecs


class PSpaceEKFAC:
    def __init__(self, generator):
        self.generator = generator
        evecs = dict()
        diags = dict()

        kfac_blocks = generator.get_kfac_blocks()
        for layer_id, layer in self.generator.layer_collection.layers.items():
            a, g = kfac_blocks[layer_id]
            evals_a, evecs_a = torch.symeig(a, eigenvectors=True)
            evals_g, evecs_g = torch.symeig(g, eigenvectors=True)
            evecs[layer_id] = (evecs_a, evecs_g)
            diags[layer_id] = kronecker(evals_g.view(-1, 1),
                                        evals_a.view(-1, 1))
        self.data = (evecs, diags)

    def get_dense_tensor(self, split_weight_bias=True):
        """
        - split_weight_bias (bool): if True then the parameters are ordered in
        the same way as in the dense or blockdiag representation, but it
        involves more operations. Otherwise the coefficients corresponding
        to the bias are mixed between coefficients of the weight matrix
        """
        evecs, diags = self.data
        s = self.generator.layer_collection.numel()
        M = torch.zeros((s, s), device=self.generator.get_device())
        for layer_id, layer in self.generator.layer_collection.layers.items():
            evecs_a, evecs_g = evecs[layer_id]
            diag = diags[layer_id]
            start = self.generator.layer_collection.p_pos[layer_id]
            sAG = diag.numel()
            if split_weight_bias:
                kronecker(evecs_g, evecs_a[:-1, :])
                kronecker(evecs_g, evecs_a[-1:, :].contiguous())
                KFE = torch.cat([kronecker(evecs_g, evecs_a[:-1, :]),
                                 kronecker(evecs_g, evecs_a[-1:, :])], dim=0)
            else:
                KFE = kronecker(evecs_g, evecs_a)
            M[start:start+sAG, start:start+sAG].add_(
                    torch.mm(KFE, torch.mm(torch.diag(diag.view(-1)),
                                           KFE.t())))
        return M

    def update_diag(self):
        self.data = (self.data[0], self.generator.get_kfe_diag(self.data[0]))

    def mv(self, vs):
        vs_dict = vs.get_dict_representation()
        out_dict = dict()
        evecs, diags = self.data
        for l_id, l in self.generator.layer_collection.layers.items():
            diag = diags[l_id]
            evecs_a, evecs_g = evecs[l_id]
            v = vs_dict[l_id][0].view(vs_dict[l_id][0].size(0), -1)
            if l.bias is not None:
                v = torch.cat([v, vs_dict[l_id][1].unsqueeze(1)], dim=1)
            v_kfe = torch.mm(torch.mm(evecs_g.t(), v), evecs_a)
            mv_kfe = v_kfe * diag.view(*v_kfe.size())
            mv = torch.mm(torch.mm(evecs_g, mv_kfe), evecs_a.t())
            if l.bias is None:
                mv_tuple = (mv,)
            else:
                mv_tuple = (mv[:, :-1].contiguous(), mv[:, -1].contiguous())
            out_dict[l_id] = mv_tuple
        return PVector(layer_collection=vs.layer_collection,
                       dict_repr=out_dict)

    def vTMv(self, vector):
        vector_dict = vector.get_dict_representation()
        evecs, diags = self.data
        norm2 = 0
        for l_id in vector_dict.keys():
            evecs_a, evecs_g = evecs[l_id]
            diag = diags[l_id]
            v = vector_dict[l_id][0].view(vector_dict[l_id][0].size(0), -1)
            if len(vector_dict[l_id]) > 1:
                v = torch.cat([v, vector_dict[l_id][1].unsqueeze(1)], dim=1)

            v_kfe = torch.mm(torch.mm(evecs_g.t(), v), evecs_a)
            norm2 += torch.dot(v_kfe.view(-1)**2, diag.view(-1))
        return norm2

    def trace(self):
        return sum([d.sum() for d in self.data[1].values()])

    def frobenius_norm(self):
        return sum([(d**2).sum() for d in self.data[1].values()])**.5


class PSpaceImplicit(PSpaceAbstract):
    def __init__(self, generator):
        self.generator = generator

    def mv(self, v):
        return self.generator.implicit_mv(v)

    def vTMv(self, v):
        return self.generator.implicit_vTMv(v)

    def trace(self):
        return self.generator.implicit_trace()

    def size(self, dim=None):
        s = self.generator.layer_collection.numel()
        if dim == 0 or dim == 1:
            return s
        elif dim is None:
            return (s, s)
        else:
            raise IndexError

    def frobenius_norm(self):
        raise NotImplementedError

    def get_dense_tensor(self):
        raise NotImplementedError

    def inverse(self, regul):
        raise NotImplementedError

    def get_diag(self):
        raise NotImplementedError


class PSpaceLowRank(PSpaceAbstract):
    def __init__(self, generator, data=None):
        self.generator = generator
        if data is not None:
            self.data = data
        else:
            self.data = generator.get_jacobian()
            self.data /= self.data.size(1)**.5

    def vTMv(self, v):
        data_mat = self.data.view(-1, self.data.size(2))
        Av = torch.mv(data_mat, v.get_flat_representation())
        return torch.dot(Av, Av)

    def get_dense_tensor(self):
        # you probably don't want to do that: you are
        # loosing the benefit of having a low rank representation
        # of your matrix but instead compute the potentially
        # much larger dense matrix
        return torch.mm(self.data.view(-1, self.data.size(2)).t(),
                        self.data.view(-1, self.data.size(2)))

    def mv(self, v):
        data_mat = self.data.view(-1, self.data.size(2))
        v_flat = torch.mv(data_mat.t(),
                          torch.mv(data_mat, v.get_flat_representation()))
        return PVector(v.layer_collection, vector_repr=v_flat)

    def compute_eigendecomposition(self, impl='symeig'):
        if impl == 'symeig':
            self.evals, V = torch.symeig(torch.mm(self.data, self.data.t()),
                                         eigenvectors=True)
            self.evecs = torch.mm(self.data.t(), V) / \
                (self.evals**.5).unsqueeze(0)
        else:
            raise NotImplementedError

    def get_eigendecomposition(self):
        return self.evals, self.evecs

    def trace(self):
        A = torch.mm(self.data.view(-1, self.data.size(2)),
                     self.data.view(-1, self.data.size(2)).t())
        return torch.trace(A)

    def frobenius_norm(self):
        A = torch.mm(self.data.view(-1, self.data.size(2)),
                     self.data.view(-1, self.data.size(2)).t())
        return torch.norm(A)

    def inverse(self, regul):
        raise NotImplementedError

    def get_diag(self):
        return (self.data**2).sum(dim=(0, 1))

    def __rmul__(self, x):
        return PSpaceLowRank(generator=self.generator,
                             data=x**.5 * self.data)


class KrylovLowRankMatrix(PSpaceAbstract):
    def __init__(self, generator):
        raise NotImplementedError()
