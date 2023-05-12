import torch
from abc import ABC, abstractmethod
from ..maths import kronecker
from .vector import PVector
from nngeometry.generator.dummy import DummyGenerator


class PMatAbstract(ABC):
    """
    A :math:`d \\times d` matrix in parameter space. This abstract class
    defines common methods used in concrete representations.

    :param generator: The generator
    :type generator: :class:`nngeometry.generator.jacobian.Jacobian`
    :param data: if None, it requires examples to be different from None, and
        it uses the generator to populate the matrix data
    :param examples: if data is None, it uses these examples to populate the
        matrix using the generator. `examples` is either a Dataloader, or a
        single mini-batch of (inputs, targets) from a Dataloader

    Note:
        Either `data` or `examples` has to be different from None, and both
        cannot be not None at the same time.
    """

    @abstractmethod
    def __init__(self, generator, data=None, examples=None):
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
        """
        Computes the quadratic form defined by M in v,
        namely the product :math:`v^\\top M v`

        :param v: vector :math:`v`
        :type v: :class:`.object.vector.PVector`
        """
        raise NotImplementedError

    @abstractmethod
    def solve(self, v, regul):
        """
        Solves Fx = v in x

        :param regul: Tikhonov regularization
        :type regul: float
        :param v: v
        :type regul: PVector
        """
        raise NotImplementedError

    @abstractmethod
    def get_diag(self):
        """
        Computes and returns the diagonal elements of this matrix.

        :return: a PyTorch Tensor
        """
        raise NotImplementedError

    def size(self, dim=None):
        """
        Size of the matrix as a tuple, regardless of the actual size in memory.

        :param dim: dimension
        :type dim: int or None

        >>> M.size()
        (1254, 1254)
        >>> M.size(0)
        1254
        """
        # TODO: test
        s = self.generator.layer_collection.numel()
        if dim == 0 or dim == 1:
            return s
        elif dim is None:
            return (s, s)
        else:
            raise IndexError

    def _check_data_examples(self, data, examples):
        """
        Either data or examples has to be not None in order
        to populate the matrix. If both are not None, then
        it is ambiguous, then the following test will fail
        """
        assert (data is not None) ^ (examples is not None)

    def __getstate__(self):
        return {'layer_collection': self.generator.layer_collection,
                'data': self.data,
                'device': self.generator.get_device()}

    def __setstate__(self, state_dict):
        self.data = state_dict['data']
        self.generator = DummyGenerator(state_dict['layer_collection'],
                                        state_dict['device'])


class PMatDense(PMatAbstract):
    def __init__(self, generator, data=None, examples=None):
        self._check_data_examples(data, examples)

        self.generator = generator
        if data is not None:
            self.data = data
        else:
            self.data = generator.get_covariance_matrix(examples)

    def compute_eigendecomposition(self, impl='eigh'):
        if impl == 'eigh':
            self.evals, self.evecs = torch.linalg.eigh(self.data)
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
            inv_v = torch.linalg.solve(self.data + 
                                       regul * torch.eye(self.size(0),
                                                         device=self.data.device),
                                       v.get_flat_representation().view(-1, 1))
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
        return PMatDense(generator=self.generator,
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

    def trace(self):
        return torch.trace(self.data)

    def get_dense_tensor(self):
        return self.data

    def get_diag(self):
        return torch.diag(self.data)

    def __add__(self, other):
        sum_data = self.data + other.data
        return PMatDense(generator=self.generator,
                         data=sum_data)

    def __sub__(self, other):
        sub_data = self.data - other.data
        return PMatDense(generator=self.generator,
                         data=sub_data)

    def __rmul__(self, x):
        return PMatDense(generator=self.generator,
                         data=x * self.data)

    def mm(self, other):
        """
        Matrix-matrix product where `other` is another 
        instance of PMatDense

        :param other: Other FIM matrix
        :type other: :class:`nngeometry.object.PMatDense`

        :return: The matrix-matrix product
        :rtype: :class:`nngeometry.object.PMatDense`
        """
        return PMatDense(self.generator,
                         data=torch.mm(self.data, other.data))


class PMatDiag(PMatAbstract):
    def __init__(self, generator, data=None, examples=None):
        self._check_data_examples(data, examples)

        self.generator = generator
        if data is not None:
            self.data = data
        else:
            self.data = generator.get_covariance_diag(examples)

    def inverse(self, regul=1e-8):
        inv_tensor = 1. / (self.data + regul)
        return PMatDiag(generator=self.generator,
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

    def solve(self, v, regul=1e-8):
        """
        solves v = Ax in x
        """
        # TODO: test
        solution = v.get_flat_representation() / (self.data + regul)
        return PVector(layer_collection=v.layer_collection,
                       vector_repr=solution)

    def __add__(self, other):
        sum_diags = self.data + other.data
        return PMatDiag(generator=self.generator,
                        data=sum_diags)

    def __sub__(self, other):
        sub_diags = self.data - other.data
        return PMatDiag(generator=self.generator,
                        data=sub_diags)

    def __rmul__(self, x):
        return PMatDiag(generator=self.generator,
                        data=x * self.data)

    def mm(self, other):
        """
        Matrix-matrix product where `other` is another 
        instance of PMatDiag

        :param other: Other FIM matrix
        :type other: :class:`nngeometry.object.PMatDiag`

        :return: The matrix-matrix product
        :rtype: :class:`nngeometry.object.PMatDiag`
        """
        return PMatDiag(self.generator,
                         data=self.data * other.data)


class PMatBlockDiag(PMatAbstract):
    def __init__(self, generator, data=None, examples=None):
        self._check_data_examples(data, examples)

        self.generator = generator
        if data is not None:
            self.data = data
        else:
            self.data = generator.get_covariance_layer_blocks(examples)

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

    def solve(self, vs, regul=1e-8):
        vs_dict = vs.get_dict_representation()
        out_dict = dict()
        for layer_id, layer in self.generator.layer_collection.layers.items():
            v = vs_dict[layer_id][0].view(-1)
            if layer.bias is not None:
                v = torch.cat([v, vs_dict[layer_id][1].view(-1)])
            block = self.data[layer_id]

            inv_v = torch.linalg.solve(block +
                                       regul * torch.eye(block.size(0),
                                                         device=block.device),
                                       v.view(-1, 1))
            inv_v_tuple = (inv_v[:layer.weight.numel()]
                           .view(*layer.weight.size),)
            if layer.bias is not None:
                inv_v_tuple = (inv_v_tuple[0],
                               inv_v[layer.weight.numel():]
                               .view(*layer.bias.size),)

            out_dict[layer_id] = inv_v_tuple
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
        return PMatBlockDiag(generator=self.generator,
                             data=inv_data)

    def frobenius_norm(self):
        # TODO test
        return sum([torch.norm(b)**2 for b in self.data.values()])**.5

    def vTMv(self, vector):
        # TODO test
        vector_dict = vector.get_dict_representation()
        norm2 = 0
        for layer_id, layer in self.generator.layer_collection.layers.items():
            v = vector_dict[layer_id][0].view(-1)
            if len(vector_dict[layer_id]) > 1:
                v = torch.cat([v, vector_dict[layer_id][1].view(-1)])
            norm2 += torch.dot(torch.mv(self.data[layer_id], v), v)
        return norm2

    def __add__(self, other):
        sum_data = {l_id: d + other.data[l_id]
                    for l_id, d in self.data.items()}
        return PMatBlockDiag(generator=self.generator,
                             data=sum_data)

    def __sub__(self, other):
        sum_data = {l_id: d - other.data[l_id]
                    for l_id, d in self.data.items()}
        return PMatBlockDiag(generator=self.generator,
                             data=sum_data)

    def __rmul__(self, x):
        sum_data = {l_id: x * d for l_id, d in self.data.items()}
        return PMatBlockDiag(generator=self.generator,
                             data=sum_data)

    def mm(self, other):
        """
        Matrix-matrix product where `other` is another 
        instance of PMatBlockDiag

        :param other: Other FIM matrix
        :type other: :class:`nngeometry.object.PMatBlockDiag`

        :return: The matrix-matrix product
        :rtype: :class:`nngeometry.object.PMatBlockDiag`
        """
        prod = dict()
        for layer_id, block in self.data.items():
            block_other = other.data[layer_id]
            prod[layer_id] = torch.mm(block, block_other)
        return PMatBlockDiag(self.generator,
                             data=prod)


class PMatKFAC(PMatAbstract):
    def __init__(self, generator, data=None, examples=None):
        self._check_data_examples(data, examples)

        self.generator = generator
        if data is None:
            self.data = generator.get_kfac_blocks(examples)
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
        return PMatKFAC(generator=self.generator,
                        data=inv_data)

    def solve(self, vs, regul=1e-8, use_pi=True):
        vs_dict = vs.get_dict_representation()
        out_dict = dict()
        for layer_id, layer in self.generator.layer_collection.layers.items():
            vw = vs_dict[layer_id][0] 
            sw = vw.size()
            v = vw.view(sw[0], -1)
            if layer.bias is not None:
                v = torch.cat([v, vs_dict[layer_id][1].unsqueeze(1)], dim=1)
            a, g = self.data[layer_id]
            if use_pi:
                pi = (torch.trace(a) / torch.trace(g) *
                      g.size(0) / a.size(0))**.5
            else:
                pi = 1
            a_reg = a + regul**.5 * pi * torch.eye(a.size(0), device=g.device)
            g_reg = g + regul**.5 / pi * torch.eye(g.size(0), device=g.device)

            solve_g, _, _, _ = torch.linalg.lstsq(g_reg, v)
            solve_a, _, _, _ = torch.linalg.lstsq(a_reg, solve_g.t())
            solve_a = solve_a.t()
            if layer.bias is None:
                solve_tuple = (solve_a.view(*sw),)
            else:
                solve_tuple = (solve_a[:, :-1].contiguous().view(*sw),
                               solve_a[:, -1].contiguous())
            out_dict[layer_id] = solve_tuple
        return PVector(layer_collection=vs.layer_collection,
                       dict_repr=out_dict)

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
            vw = vs_dict[layer_id][0] 
            sw = vw.size()
            v = vw.view(sw[0], -1)
            if layer.bias is not None:
                v = torch.cat([v, vs_dict[layer_id][1].unsqueeze(1)], dim=1)
            a, g = self.data[layer_id]
            mv = torch.mm(torch.mm(g, v), a)
            if layer.bias is None:
                mv_tuple = (mv.view(*sw),)
            else:
                mv_tuple = (mv[:, :-1].contiguous().view(*sw),
                            mv[:, -1].contiguous())
            out_dict[layer_id] = mv_tuple
        return PVector(layer_collection=vs.layer_collection,
                       dict_repr=out_dict)

    def vTMv(self, vector):
        vector_dict = vector.get_dict_representation()
        norm2 = 0
        for layer_id, layer in self.generator.layer_collection.layers.items():
            v = vector_dict[layer_id][0].view(vector_dict[layer_id][0].size(0),
                                              -1)
            if layer.bias is not None:
                v = torch.cat([v, vector_dict[layer_id][1].unsqueeze(1)],
                              dim=1)
            a, g = self.data[layer_id]
            norm2 += torch.dot(torch.mm(torch.mm(g, v), a).view(-1),
                               v.view(-1))
        return norm2

    def frobenius_norm(self):
        return sum([torch.trace(torch.mm(a, a)) * torch.trace(torch.mm(g, g))
                    for a, g in self.data.values()])**.5

    def compute_eigendecomposition(self, impl='eigh'):
        self.evals = dict()
        self.evecs = dict()
        if impl == 'eigh':
            for layer_id in self.generator.layer_collection.layers.keys():
                a, g = self.data[layer_id]
                evals_a, evecs_a = torch.linalg.eigh(a)
                evals_g, evecs_g = torch.linalg.eigh(g)
                self.evals[layer_id] = (evals_a, evals_g)
                self.evecs[layer_id] = (evecs_a, evecs_g)
        else:
            raise NotImplementedError

    def get_eigendecomposition(self):
        return self.evals, self.evecs

    def mm(self, other):
        """
        Matrix-matrix product where `other` is another 
        instance of PMatKFAC

        :param other: Other FIM matrix
        :type other: :class:`nngeometry.object.PMatKFAC`

        :return: The matrix-matrix product
        :rtype: :class:`nngeometry.object.PMatKFAC`
        """
        prod = dict()
        for layer_id, (a, g) in self.data.items():
            (a_other, g_other) = other.data[layer_id]
            prod[layer_id] = (torch.mm(a, a_other),
                              torch.mm(g, g_other))
        return PMatKFAC(self.generator, data=prod)


class PMatEKFAC(PMatAbstract):
    """
    EKFAC representation from
    *George, Laurent et al., Fast Approximate Natural Gradient Descent
    in a Kronecker-factored Eigenbasis, NIPS 2018*

    """
    def __init__(self, generator, data=None, examples=None):
        self._check_data_examples(data, examples)

        self.generator = generator
        if data is None:
            evecs = dict()
            diags = dict()

            kfac_blocks = generator.get_kfac_blocks(examples)
            for layer_id, layer in \
                    self.generator.layer_collection.layers.items():
                a, g = kfac_blocks[layer_id]
                evals_a, evecs_a = torch.linalg.eigh(a)
                evals_g, evecs_g = torch.linalg.eigh(g)
                evecs[layer_id] = (evecs_a, evecs_g)
                diags[layer_id] = kronecker(evals_g.view(-1, 1),
                                            evals_a.view(-1, 1))
                del a, g, kfac_blocks[layer_id]
            self.data = (evecs, diags)
        else:
            self.data = data

    def get_dense_tensor(self, split_weight_bias=True):
        """
        - split_weight_bias (bool): if True then the parameters are ordered in
        the same way as in the dense or blockdiag representation, but it
        involves more operations. Otherwise the coefficients corresponding
        to the bias are mixed between coefficients of the weight matrix
        """
        _, diags = self.data
        s = self.generator.layer_collection.numel()
        M = torch.zeros((s, s), device=self.generator.get_device())
        KFE_layers = self.get_KFE(split_weight_bias=split_weight_bias)
        for layer_id, _ in self.generator.layer_collection.layers.items():
            diag = diags[layer_id]
            start = self.generator.layer_collection.p_pos[layer_id]
            sAG = diag.numel()
            KFE = KFE_layers[layer_id]
            M[start:start+sAG, start:start+sAG].add_(
                    torch.mm(KFE, torch.mm(torch.diag(diag.view(-1)),
                                           KFE.t())))
        return M

    def get_KFE(self, split_weight_bias=True):
        """
        Returns a dict index by layers, of dense eigenvectors constructed from
        Kronecker-factored eigenvectors

        - split_weight_bias (bool): if True then the parameters are ordered in
        the same way as in the dense or blockdiag representation, but it
        involves more operations. Otherwise the coefficients corresponding
        to the bias are mixed between coefficients of the weight matrix
        """
        evecs, _ = self.data
        KFE = dict()
        for layer_id, _ in self.generator.layer_collection.layers.items():
            evecs_a, evecs_g = evecs[layer_id]
            start = self.generator.layer_collection.p_pos[layer_id]
            if split_weight_bias:
                kronecker(evecs_g, evecs_a[:-1, :])
                kronecker(evecs_g, evecs_a[-1:, :].contiguous())
                KFE[layer_id] = torch.cat([kronecker(evecs_g, evecs_a[:-1, :]),
                                 kronecker(evecs_g, evecs_a[-1:, :])], dim=0)
            else:
                KFE[layer_id] = kronecker(evecs_g, evecs_a)
        return KFE

    def update_diag(self, examples):
        """
        Will update the diagonal in the KFE (aka the approximate eigenvalues)
        using current values of the model's parameters
        """
        self.data = (self.data[0], self.generator.get_kfe_diag(self.data[0], examples))

    def mv(self, vs):
        vs_dict = vs.get_dict_representation()
        out_dict = dict()
        evecs, diags = self.data
        for l_id, l in self.generator.layer_collection.layers.items():
            diag = diags[l_id]
            evecs_a, evecs_g = evecs[l_id]
            vw = vs_dict[l_id][0] 
            sw = vw.size()
            v = vw.view(sw[0], -1)
            if l.bias is not None:
                v = torch.cat([v, vs_dict[l_id][1].unsqueeze(1)], dim=1)
            v_kfe = torch.mm(torch.mm(evecs_g.t(), v), evecs_a)
            mv_kfe = v_kfe * diag.view(*v_kfe.size())
            mv = torch.mm(torch.mm(evecs_g, mv_kfe), evecs_a.t())
            if l.bias is None:
                mv_tuple = (mv.view(*sw),)
            else:
                mv_tuple = (mv[:, :-1].contiguous().view(*sw),
                            mv[:, -1].contiguous())
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

    def get_diag(self, v):
        raise NotImplementedError

    def inverse(self, regul=1e-8):
        evecs, diags = self.data
        inv_diags = {i: 1. / (d + regul)
                     for i, d in diags.items()}
        return PMatEKFAC(generator=self.generator,
                         data=(evecs, inv_diags))

    def solve(self, vs, regul=1e-8):
        vs_dict = vs.get_dict_representation()
        out_dict = dict()
        evecs, diags = self.data
        for l_id, l in self.generator.layer_collection.layers.items():
            diag = diags[l_id]
            evecs_a, evecs_g = evecs[l_id]
            vw = vs_dict[l_id][0] 
            sw = vw.size()
            v = vw.view(sw[0], -1)
            if l.bias is not None:
                v = torch.cat([v, vs_dict[l_id][1].unsqueeze(1)], dim=1)
            v_kfe = torch.mm(torch.mm(evecs_g.t(), v), evecs_a)
            inv_kfe = v_kfe / (diag.view(*v_kfe.size()) + regul)
            inv = torch.mm(torch.mm(evecs_g, inv_kfe), evecs_a.t())
            if l.bias is None:
                inv_tuple = (inv.view(*sw),)
            else:
                inv_tuple = (inv[:, :-1].contiguous().view(*sw),
                             inv[:, -1].contiguous())
            out_dict[l_id] = inv_tuple
        return PVector(layer_collection=vs.layer_collection,
                       dict_repr=out_dict)

    def __rmul__(self, x):
        evecs, diags = self.data
        diags = {l_id: x * d for l_id, d in diags.items()}
        return PMatEKFAC(generator=self.generator,
                         data=(evecs, diags))


class PMatImplicit(PMatAbstract):
    """
    PMatImplicit is a very special representation, since
    no elements of the matrix is ever computed, but instead
    various linear algebra operations are performed implicitely
    using efficient tricks.

    The computations are done exactly, meaning that there is
    no approximation involved. This is useful for networks too big
    to fit in memory.
    """
    def __init__(self, generator, data=None, examples=None):
        self.generator = generator

        assert data is None

        self.examples = examples

    def mv(self, v):
        return self.generator.implicit_mv(v, self.examples)

    def vTMv(self, v):
        return self.generator.implicit_vTMv(v, self.examples)

    def trace(self):
        return self.generator.implicit_trace(self.examples)

    def frobenius_norm(self):
        raise NotImplementedError

    def get_dense_tensor(self):
        raise NotImplementedError

    def solve(self, v):
        raise NotImplementedError

    def get_diag(self):
        raise NotImplementedError


class PMatLowRank(PMatAbstract):
    def __init__(self, generator, data=None, examples=None):
        self._check_data_examples(data, examples)

        self.generator = generator
        if data is not None:
            self.data = data
        else:
            self.data = generator.get_jacobian(examples)
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

    def compute_eigendecomposition(self, impl='svd'):
        data_mat = self.data.view(-1, self.data.size(2))
        if impl == 'svd':
            _, sqrt_evals, self.evecs = torch.svd(data_mat, some=True)
            self.evals = sqrt_evals**2
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

    def solve(self, b, regul=1e-8):
        u, s, v = torch.svd(self.data.view(-1, self.data.size(2)))
        x = torch.mv(v, torch.mv(v.t(), b.get_flat_representation()) /
                     (s**2 + regul))
        return PVector(b.layer_collection, vector_repr=x)

    def get_diag(self):
        return (self.data**2).sum(dim=(0, 1))

    def __rmul__(self, x):
        return PMatLowRank(generator=self.generator,
                           data=x**.5 * self.data)


class PMatQuasiDiag(PMatAbstract):
    """
    Quasidiagonal approximation as decribed in Ollivier,
    Riemannian metrics for neural networks I: feedforward networks,
    Information and Inference: A Journal of the IMA, 2015
    """
    def __init__(self, generator, data=None, examples=None):
        self._check_data_examples(data, examples)

        self.generator = generator
        if data is not None:
            self.data = data
        else:
            self.data = generator.get_covariance_quasidiag(examples)

    def get_dense_tensor(self):
        s = self.generator.layer_collection.numel()
        device = self.generator.get_device()
        M = torch.zeros((s, s), device=device)
        for layer_id in self.generator.layer_collection.layers.keys():
            diag, cross = self.data[layer_id]
            block_s = diag.size(0)
            block = torch.diag(diag)
            if cross is not None:
                out_s = cross.size(0)
                in_s = cross.numel() // out_s

                block_bias = torch.cat((cross.view(cross.size(0), -1).t()
                                        .reshape(-1, 1),
                                        torch.zeros((out_s * in_s, out_s),
                                                    device=device)),
                                       dim=1)
                block_bias = block_bias.view(in_s, out_s+1, out_s) \
                    .transpose(0, 1).reshape(-1, out_s)[:in_s*out_s, :]

                block[:in_s*out_s, in_s*out_s:].copy_(block_bias)
                block[in_s*out_s:, :in_s*out_s].copy_(block_bias.t())
            start = self.generator.layer_collection.p_pos[layer_id]
            M[start:start+block_s, start:start+block_s].add_(block)
        return M

    def frobenius_norm(self):
        norm2 = 0
        for layer_id in self.generator.layer_collection.layers.keys():
            diag, cross = self.data[layer_id]
            norm2 += torch.dot(diag, diag)
            if cross is not None:
                norm2 += 2 * torch.dot(cross.view(-1), cross.view(-1))

        return norm2 ** .5

    def get_diag(self):
        return torch.cat([self.data[l_id][0] for l_id in
                          self.generator.layer_collection.layers.keys()])

    def trace(self):
        return sum([self.data[l_id][0].sum() for l_id in
                    self.generator.layer_collection.layers.keys()])

    def vTMv(self, vs):
        vs_dict = vs.get_dict_representation()
        out = 0
        for layer_id, layer in self.generator.layer_collection.layers.items():
            diag, cross = self.data[layer_id]
            v_weight = vs_dict[layer_id][0]
            if layer.bias is not None:
                v_bias = vs_dict[layer_id][1]
            mv_bias = None
            mv_weight = diag[:layer.weight.numel()] * v_weight.view(-1)
            if layer.bias is not None:
                mv_bias = diag[layer.weight.numel():] * v_bias.view(-1)
                mv_bias += (cross * v_weight).view(v_bias.size(0), -1) \
                    .sum(dim=1)
                if len(cross.size()) == 2:
                    mv_weight += (cross * v_bias.view(-1, 1)).view(-1)
                elif len(cross.size()) == 4:
                    mv_weight += (cross * v_bias.view(-1, 1, 1, 1)).view(-1)
                else:
                    raise NotImplementedError
                out += torch.dot(mv_bias, v_bias)

            out += torch.dot(mv_weight, v_weight.view(-1))
        return out

    def mv(self, vs):
        vs_dict = vs.get_dict_representation()
        out_dict = dict()
        for layer_id, layer in self.generator.layer_collection.layers.items():
            diag, cross = self.data[layer_id]

            v_weight = vs_dict[layer_id][0]
            if layer.bias is not None:
                v_bias = vs_dict[layer_id][1]

            mv_bias = None
            mv_weight = diag[:layer.weight.numel()].view(*v_weight.size()) \
                * v_weight
            if layer.bias is not None:
                mv_bias = diag[layer.weight.numel():] * v_bias.view(-1)
                mv_bias += (cross * v_weight).view(v_bias.size(0), -1) \
                    .sum(dim=1)
                if len(cross.size()) == 2:
                    mv_weight += cross * v_bias.view(-1, 1)
                elif len(cross.size()) == 4:
                    mv_weight += cross * v_bias.view(-1, 1, 1, 1)
                else:
                    raise NotImplementedError

            out_dict[layer_id] = (mv_weight, mv_bias)
        return PVector(layer_collection=vs.layer_collection,
                       dict_repr=out_dict)

    def solve(self, vs, regul=1e-8):
        vs_dict = vs.get_dict_representation()
        out_dict = dict()
        for layer_id, layer in self.generator.layer_collection.layers.items():
            diag, cross = self.data[layer_id]

            v_weight = vs_dict[layer_id][0]
            # keep original size
            s_w = v_weight.size()
            v_weight = v_weight.view(s_w[0], -1)

            d_weight = diag[:layer.weight.numel()].view(s_w[0], -1) + regul
            solve_b = None
            if layer.bias is None:
                solve_w = v_weight / d_weight
            else:
                v_bias = vs_dict[layer_id][1]
                d_bias = diag[layer.weight.numel():] + regul

                cross = cross.view(s_w[0], -1)

                solve_b_denom = d_bias - bdot(cross / d_weight, cross)
                solve_b = ((v_bias - bdot(cross / d_weight, v_weight))
                           / solve_b_denom)

                solve_w = (v_weight - solve_b.unsqueeze(1) * cross) / d_weight

            out_dict[layer_id] = (solve_w.view(*s_w), solve_b)
        return PVector(layer_collection=vs.layer_collection,
                       dict_repr=out_dict)


def bdot(A, B):
    """
    batched dot product
    """
    return torch.matmul(A.unsqueeze(1), B.unsqueeze(2)).squeeze(1).squeeze(1)