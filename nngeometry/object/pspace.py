from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
import warnings

import torch

from nngeometry.backend import DummyGenerator
from nngeometry.layercollection import (
    Conv1dLayer,
    Conv2dLayer,
    EmbeddingLayer,
    LayerCollection,
    LinearLayer,
)
from nngeometry.object.map import PFMap, PFMapDense

from ..maths import kronecker
from .vector import PVector


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
    def __init__(self, layer_collection, generator, data=None, examples=None, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def to_torch(self):
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
    def get_device(self):
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
    def solvePVec(self, x, regul, solve):
        raise NotImplementedError

    def solvePFMap(self, x, regul, solve):
        J_dense = x.to_torch()
        sJ = J_dense.size()
        J_dense = J_dense.view(sJ[0] * sJ[1], sJ[2])

        vs_solve = []
        for i in range(J_dense.size(0)):
            v = PVector(
                layer_collection=x.layer_collection,
                vector_repr=J_dense[i, :],
            )
            v_solve = self.solvePVec(v, regul=regul, solve=solve)
            vs_solve.append(v_solve.to_torch())
        return PFMapDense(
            generator=self.generator,
            data=torch.stack(vs_solve).view(*sJ),
            layer_collection=x.layer_collection,
        )

    def solve(self, x, regul, solve="default"):
        """
        Solves Fx = b in x

        :param regul: Tikhonov regularization
        :type regul: float
        :param b: b
        :type b: PVector or PFMap
        """
        if isinstance(x, PVector):
            return self.solvePVec(x, regul=regul, solve=solve)
        elif isinstance(x, PFMap):
            return self.solvePFMap(x, regul=regul, solve=solve)

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
        s = self.layer_collection.numel()
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
        return {
            "layer_collection": self.layer_collection,
            "data": self.data,
            "device": self.get_device(),
        }

    def __setstate__(self, state_dict):
        self.data = state_dict["data"]
        self.layer_collection = state_dict["layer_collection"]
        self.generator = DummyGenerator(state_dict["device"])


class PMatDense(PMatAbstract):
    def __init__(self, layer_collection, generator, data=None, examples=None, **kwargs):
        self._check_data_examples(data, examples)

        self.layer_collection = layer_collection
        self.generator = generator
        if data is not None:
            self.data = data
        else:
            self.data = generator.get_covariance_matrix(
                examples, layer_collection=layer_collection
            )

    def compute_eigendecomposition(self, impl="eigh"):
        if impl == "eigh":
            self.evals, self.evecs = torch.linalg.eigh(self.data)
        elif impl == "svd":
            _, self.evals, self.evecs = torch.svd(self.data, some=False)
        else:
            raise NotImplementedError

    def solvePVec(self, x, regul=1e-8, solve="solve"):
        """
        solves v = Ax in x
        """
        # TODO: test
        if solve in ["default", "solve"]:
            # TODO: reuse LU decomposition once it is computed
            inv_v = self._solve_cached(x.to_torch().view(1, -1), regul=regul)
            return PVector(x.layer_collection, vector_repr=inv_v[0, :])
        elif solve == "eigendecomposition":
            v_eigenbasis = self.project_to_diag(x)
            inv_v_eigenbasis = v_eigenbasis / (self.evals + regul)
            return self.project_from_diag(inv_v_eigenbasis)
        else:
            raise NotImplementedError

    def solvePFMap(self, x, regul=1e-8, solve="solve"):
        """
        solves J = AX in X
        """
        if solve in ["default", "solve"]:
            J_torch = x.to_torch()
            sJ = J_torch.size()
            inv_v = self._solve_cached(J_torch.view(-1, sJ[-1]), regul=regul)
            return PFMapDense(
                generator=self.generator,
                data=inv_v.reshape(*sJ),
                layer_collection=self.layer_collection,
            )
        else:
            raise NotImplementedError

    def _solve_cached(self, x, regul):
        try:  # check for cache
            assert self._ldl_regul == regul
            LD, pivots = self._ldl_factors
        except (
            AttributeError,
            AssertionError,
        ):  # ldl decomposition is not currently cached for regul value
            LD, pivots = torch.linalg.ldl_factor(
                self.data + regul * torch.eye(self.size(0), device=self.get_device()),
            )
            self._ldl_regul = regul
            self._ldl_factors = (LD, pivots)
        return torch.linalg.ldl_solve(LD, pivots, x.t()).t()

    def inverse(self, regul=1e-8):
        inv_tensor = torch.inverse(
            self.data + regul * torch.eye(self.size(0), device=self.get_device())
        )
        return PMatDense(
            generator=self.generator,
            data=inv_tensor,
            layer_collection=self.layer_collection,
        )

    def get_device(self):
        return self.data.device

    def mv(self, v):
        v_flat = torch.mv(self.data, v.to_torch())
        return PVector(v.layer_collection, vector_repr=v_flat)

    def vTMv(self, v):
        v_flat = v.to_torch()
        return torch.dot(v_flat, torch.mv(self.data, v_flat))

    def frobenius_norm(self):
        return torch.norm(self.data)

    def project_to_diag(self, v):
        # TODO: test
        return torch.mv(self.evecs.t(), v.to_torch())

    def project_from_diag(self, v):
        # TODO: test
        return PVector(
            layer_collection=self.layer_collection,
            vector_repr=torch.mv(self.evecs, v),
        )

    def get_eigendecomposition(self):
        # TODO: test
        return self.evals, self.evecs

    def trace(self):
        return torch.trace(self.data)

    def to_torch(self):
        return self.data

    def get_diag(self):
        return torch.diag(self.data)

    def __add__(self, other):
        sum_data = self.data + other.data
        return PMatDense(
            generator=self.generator,
            data=sum_data,
            layer_collection=self.layer_collection,
        )

    def __sub__(self, other):
        sub_data = self.data - other.data
        return PMatDense(
            generator=self.generator,
            data=sub_data,
            layer_collection=self.layer_collection,
        )

    def __rmul__(self, x):
        return PMatDense(
            generator=self.generator,
            data=x * self.data,
            layer_collection=self.layer_collection,
        )

    def mm(self, other):
        """
        Matrix-matrix product where `other` is another
        instance of PMatDense

        :param other: Other FIM matrix
        :type other: :class:`nngeometry.object.PMatDense`

        :return: The matrix-matrix product
        :rtype: :class:`nngeometry.object.PMatDense`
        """
        return PMatDense(
            generator=self.generator,
            data=torch.mm(self.data, other.data),
            layer_collection=self.layer_collection,
        )


class PMatDiag(PMatAbstract):
    def __init__(self, layer_collection, generator, data=None, examples=None, **kwargs):
        self._check_data_examples(data, examples)

        self.layer_collection = layer_collection
        self.generator = generator
        if data is not None:
            self.data = data
        else:
            self.data = generator.get_covariance_diag(
                examples, layer_collection=layer_collection
            )

    def inverse(self, regul=1e-8):
        inv_tensor = 1.0 / (self.data + regul)
        return PMatDiag(
            generator=self.generator,
            data=inv_tensor,
            layer_collection=self.layer_collection,
        )

    def get_device(self):
        return self.data.device

    def mv(self, v):
        v_flat = v.to_torch() * self.data
        return PVector(v.layer_collection, vector_repr=v_flat)

    def trace(self):
        return self.data.sum()

    def vTMv(self, v):
        v_flat = v.to_torch()
        return torch.dot(v_flat, self.data * v_flat)

    def frobenius_norm(self):
        return torch.norm(self.data)

    def to_torch(self):
        return torch.diag(self.data)

    def get_diag(self):
        return self.data

    def solvePVec(self, x, regul=1e-8, solve="default"):
        """
        solves v = Ax in x
        """
        # TODO: test
        if solve != "default":
            raise NotImplementedError
        solution = x.to_torch() / (self.data + regul)
        return PVector(layer_collection=x.layer_collection, vector_repr=solution)

    def __add__(self, other):
        sum_diags = self.data + other.data
        return PMatDiag(
            generator=self.generator,
            data=sum_diags,
            layer_collection=self.layer_collection,
        )

    def __sub__(self, other):
        sub_diags = self.data - other.data
        return PMatDiag(
            generator=self.generator,
            data=sub_diags,
            layer_collection=self.layer_collection,
        )

    def __rmul__(self, x):
        return PMatDiag(
            generator=self.generator,
            data=x * self.data,
            layer_collection=self.layer_collection,
        )

    def mm(self, other):
        """
        Matrix-matrix product where `other` is another
        instance of PMatDiag

        :param other: Other FIM matrix
        :type other: :class:`nngeometry.object.PMatDiag`

        :return: The matrix-matrix product
        :rtype: :class:`nngeometry.object.PMatDiag`
        """
        return PMatDiag(
            generator=self.generator,
            data=self.data * other.data,
            layer_collection=self.layer_collection,
        )


class PMatBlockDiag(PMatAbstract):
    def __init__(self, layer_collection, generator, data=None, examples=None, **kwargs):
        self._check_data_examples(data, examples)

        self.layer_collection = layer_collection
        self.generator = generator
        if data is not None:
            self.data = data
        else:
            self.data = generator.get_covariance_layer_blocks(
                examples, layer_collection=layer_collection
            )

    def trace(self):
        # TODO test
        return sum([torch.trace(b) for b in self.data.values()])

    def get_device(self):
        return next(iter(self.data.values())).device

    def get_block_torch(self, layer_id, layer):
        return self.data[layer_id]

    def to_torch(self):
        s = self.layer_collection.numel()
        M = torch.zeros((s, s), device=self.get_device())
        for layer_id, layer in self.layer_collection.layers.items():
            start = self.layer_collection.p_pos[layer_id]
            numel = layer.numel()
            M[start : start + numel, start : start + numel].add_(
                self.get_block_torch(layer_id, layer)
            )
        return M

    def get_diag(self):
        diag = []
        for layer_id in self.layer_collection.layers.keys():
            b = self.data[layer_id]
            diag.append(torch.diag(b))
        return torch.cat(diag)

    def mv(self, vs):
        vs_dict = vs.to_dict()
        out_dict = dict()
        lc_merged = self.layer_collection.merge(vs.layer_collection)
        for layer_id, layer in lc_merged.layers.items():
            v = vs_dict[layer_id][0].view(-1)
            if layer.bias is not None:
                v = torch.cat([v, vs_dict[layer_id][1].view(-1)])
            mv = torch.mv(self.data[layer_id], v)
            mv_tuple = (mv[: layer.weight.numel()].view(*layer.weight.size),)
            if layer.bias is not None:
                mv_tuple = (
                    mv_tuple[0],
                    mv[layer.weight.numel() :].view(*layer.bias.size),
                )
            out_dict[layer_id] = mv_tuple
        return PVector(layer_collection=lc_merged, dict_repr=out_dict)

    def solve_block(self, d, layer_id, regul, solve):
        if callable(solve):
            solve_fn = solve
        elif solve not in ["solve", "default"]:
            raise NotImplementedError
        else:
            solve_fn = torch.linalg.solve

        layer = self.layer_collection.layers[layer_id]
        v = d[0].view(-1, layer.weight.numel())
        if layer.has_bias():
            v = torch.cat([v, d[1].view(-1, layer.bias.numel())], dim=1)
        block = self.data[layer_id]

        inv_v = solve_fn(
            block
            + regul
            * torch.eye(block.size(0), device=self.get_device(), dtype=block.dtype),
            v.t(),
        ).t()
        inv_v_tuple = (inv_v[:, : layer.weight.numel()].view(-1, *layer.weight.size),)
        if layer.has_bias():
            inv_v_tuple = (
                inv_v_tuple[0],
                inv_v[:, layer.weight.numel() :].reshape(-1, *layer.bias.size),
            )

        return inv_v_tuple

    def solvePVec(self, x, regul=1e-8, solve="solve"):
        out_dict = dict()
        lc_merged = self.layer_collection.merge(x.layer_collection)
        for layer_id in lc_merged.layers.keys():
            d = x.to_torch_layer(layer_id)
            out_dict[layer_id] = tuple(
                p[0] for p in self.solve_block(d, layer_id, regul=regul, solve=solve)
            )
        return PVector(layer_collection=lc_merged, dict_repr=out_dict)

    def solvePFMap(self, x, regul=1e-8, solve="solve"):
        out_dict = dict()
        lc_merged = self.layer_collection.merge(x.layer_collection)
        for layer_id in lc_merged.layers.keys():
            d = x.to_torch_layer(layer_id)
            so, sb, *_ = d[0].size()  # out x minibatch x rest
            out_dict[layer_id] = tuple(
                p.view(so, sb, -1)
                for p in self.solve_block(d, layer_id, regul=regul, solve=solve)
            )
        return PFMapDense.from_dict(
            layer_collection=lc_merged, generator=self.generator, data_dict=out_dict
        )

    def inverse(self, regul=1e-8):
        inv_data = dict()
        for layer_id, layer in self.layer_collection.layers.items():
            b = self.data[layer_id]
            inv_b = torch.inverse(
                b + regul * torch.eye(b.size(0), device=self.get_device())
            )
            inv_data[layer_id] = inv_b
        return PMatBlockDiag(
            generator=self.generator,
            data=inv_data,
            layer_collection=self.layer_collection,
        )

    def frobenius_norm(self):
        # TODO test
        return sum([torch.norm(b) ** 2 for b in self.data.values()]) ** 0.5

    def vTMv(self, vector):
        # TODO test
        vector_dict = vector.to_dict()
        norm2 = 0
        lc_merged = self.layer_collection.merge(vector.layer_collection)
        for layer_id, layer in lc_merged.layers.items():
            v = vector_dict[layer_id][0].view(-1)
            if len(vector_dict[layer_id]) > 1:
                v = torch.cat([v, vector_dict[layer_id][1].view(-1)])
            norm2 += torch.dot(torch.mv(self.data[layer_id], v), v)
        return norm2

    def __add__(self, other):
        sum_data = {l_id: d + other.data[l_id] for l_id, d in self.data.items()}
        return PMatBlockDiag(
            generator=self.generator,
            data=sum_data,
            layer_collection=self.layer_collection,
        )

    def __sub__(self, other):
        sum_data = {l_id: d - other.data[l_id] for l_id, d in self.data.items()}
        return PMatBlockDiag(
            generator=self.generator,
            data=sum_data,
            layer_collection=self.layer_collection,
        )

    def __rmul__(self, x):
        sum_data = {l_id: x * d for l_id, d in self.data.items()}
        return PMatBlockDiag(
            generator=self.generator,
            data=sum_data,
            layer_collection=self.layer_collection,
        )

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
        return PMatBlockDiag(
            generator=self.generator, data=prod, layer_collection=self.layer_collection
        )


class PMatKFAC(PMatAbstract):
    def __init__(self, layer_collection, generator, data=None, examples=None, **kwargs):
        self._check_data_examples(data, examples)

        self.layer_collection = layer_collection
        self.generator = generator
        if data is None:
            self.data = generator.get_kfac_blocks(
                examples, layer_collection=layer_collection
            )
        else:
            self.data = data

    def trace(self):
        return sum([torch.trace(a) * torch.trace(g) for a, g in self.data.values()])

    def inverse(self, regul=1e-8, use_pi=True):
        inv_data = dict()
        for layer_id, layer in self.layer_collection.layers.items():
            a, g = self.data[layer_id]
            if use_pi:
                pi = (torch.trace(a) / torch.trace(g) * g.size(0) / a.size(0)) ** 0.5
            else:
                pi = 1
            inv_a = torch.inverse(
                a + pi * regul**0.5 * torch.eye(a.size(0), device=self.get_device())
            )
            inv_g = torch.inverse(
                g + regul**0.5 / pi * torch.eye(g.size(0), device=self.get_device())
            )
            inv_data[layer_id] = (inv_a, inv_g)
        return PMatKFAC(
            generator=self.generator,
            data=inv_data,
            layer_collection=self.layer_collection,
        )

    def pow(self, pow, regul=1e-8, use_pi=True):
        pow_data = dict()
        for layer_id, layer in self.layer_collection.layers.items():
            a, g = self.data[layer_id]
            if use_pi:
                pi = (torch.trace(a) / torch.trace(g) * g.size(0) / a.size(0)) ** 0.5
            else:
                pi = 1
            pow_a = torch.linalg.matrix_power(
                a + pi * regul**0.5 * torch.eye(a.size(0), device=self.get_device()),
                pow,
            )
            pow_g = torch.linalg.matrix_power(
                g + regul**0.5 / pi * torch.eye(g.size(0), device=self.get_device()),
                pow,
            )
            pow_data[layer_id] = (pow_a, pow_g)
        return PMatKFAC(
            generator=self.generator,
            data=pow_data,
            layer_collection=self.layer_collection,
        )

    def __pow__(self, pow):
        return self.pow(pow)

    def solvePVec(self, x, regul=1e-8, solve="default", use_pi=True):
        if solve != "default":
            raise NotImplementedError

        vs_dict = x.to_dict()
        out_dict = dict()
        for layer_id, layer in self.layer_collection.layers.items():
            vw = vs_dict[layer_id][0]
            sw = vw.size()
            v = vw.view(sw[0], -1)
            if layer.has_bias():
                v = torch.cat([v, vs_dict[layer_id][1].unsqueeze(1)], dim=1)
            a, g = self.data[layer_id]
            if layer.transposed:
                a, g = g, a
            if use_pi:
                pi = (torch.trace(a) / torch.trace(g) * g.size(0) / a.size(0)) ** 0.5
            else:
                pi = 1
            a_reg = a + regul**0.5 * pi * torch.eye(a.size(0), device=self.get_device())
            g_reg = g + regul**0.5 / pi * torch.eye(g.size(0), device=self.get_device())

            solve_g, _, _, _ = torch.linalg.lstsq(g_reg, v)
            solve_a, _, _, _ = torch.linalg.lstsq(a_reg, solve_g.t())
            solve_a = solve_a.t()
            if layer.has_bias():
                solve_tuple = (
                    solve_a[:, :-1].contiguous().view(*sw),
                    solve_a[:, -1].contiguous(),
                )
            else:
                solve_tuple = (solve_a.view(*sw),)
            out_dict[layer_id] = solve_tuple
        return PVector(layer_collection=x.layer_collection, dict_repr=out_dict)

    def get_block_torch(self, layer_id, layer, split_weight_bias=True):
        a, g = self.data[layer_id]
        if split_weight_bias and layer.has_bias():
            block = torch.cat(
                [
                    torch.cat(
                        [
                            kronecker(g, a[:-1, :-1], transpose=layer.transposed),
                            kronecker(g, a[:-1, -1:], transpose=layer.transposed),
                        ],
                        dim=1,
                    ),
                    torch.cat(
                        [
                            kronecker(g, a[-1:, :-1], transpose=layer.transposed),
                            kronecker(g, a[-1:, -1:], transpose=layer.transposed),
                        ],
                        dim=1,
                    ),
                ],
                dim=0,
            )
        else:
            block = kronecker(g, a, transpose=layer.transposed)
        return block

    def to_torch(self, split_weight_bias=True):
        """
        - split_weight_bias (bool): if True then the parameters are ordered in
        the same way as in the dense or blockdiag representation, but it
        involves more operations. Otherwise the coefficients corresponding
        to the bias are mixed between coefficients of the weight matrix
        """
        s = self.layer_collection.numel()
        M = torch.zeros((s, s), device=self.get_device())
        for layer_id, layer in self.layer_collection.layers.items():
            a, g = self.data[layer_id]
            start = self.layer_collection.p_pos[layer_id]
            sAG = a.size(0) * g.size(0)
            M[start : start + sAG, start : start + sAG].add_(
                self.get_block_torch(
                    layer_id, layer, split_weight_bias=split_weight_bias
                )
            )
        return M

    def get_device(self):
        return next(iter(self.data.values()))[0].device

    def get_diag(self, split_weight_bias=True):
        """
        - split_weight_bias (bool): if True then the parameters are ordered in
        the same way as in the dense or blockdiag representation, but it
        involves more operations. Otherwise the coefficients corresponding
        to the bias are mixed between coefficients of the weight matrix
        """
        diags = []
        for layer_id, layer in self.layer_collection.layers.items():
            a, g = self.data[layer_id]
            if layer.transposed:
                a, g = g, a
            diag_of_block = torch.diag(g).view(-1, 1) * torch.diag(a).view(1, -1)
            if split_weight_bias and layer.has_bias():
                diags.append(diag_of_block[:, :-1].contiguous().view(-1))
                diags.append(diag_of_block[:, -1:].view(-1))
            else:
                diags.append(diag_of_block.view(-1))
        return torch.cat(diags)

    def mv(self, vs):
        vs_dict = vs.to_dict()
        out_dict = dict()
        for layer_id in (
            self.layer_collection.layers.keys() & vs_dict.keys()
        ):  # common keys
            layer = self.layer_collection.layers[layer_id]
            vw = vs_dict[layer_id][0]
            sw = vw.size()
            v = vw.view(sw[0], -1)
            if layer.has_bias():
                v = torch.cat([v, vs_dict[layer_id][1].unsqueeze(1)], dim=1)
            a, g = self.data[layer_id]
            if layer.transposed:
                a, g = g, a
            mv = torch.mm(torch.mm(g, v), a)
            if layer.has_bias():
                mv_tuple = (mv[:, :-1].contiguous().view(*sw), mv[:, -1].contiguous())
            else:
                mv_tuple = (mv.view(*sw),)
            out_dict[layer_id] = mv_tuple
        return PVector(layer_collection=vs.layer_collection, dict_repr=out_dict)

    def vTMv(self, vector):
        vector_dict = vector.to_dict()
        norm2 = 0
        for layer_id, layer in self.layer_collection.layers.items():
            v = vector_dict[layer_id][0].view(vector_dict[layer_id][0].size(0), -1)
            if layer.has_bias():
                v = torch.cat([v, vector_dict[layer_id][1].unsqueeze(1)], dim=1)
            a, g = self.data[layer_id]
            if layer.transposed:
                a, g = g, a
            norm2 += torch.dot(torch.mm(torch.mm(g, v), a).view(-1), v.view(-1))
        return norm2

    def frobenius_norm(self):
        return (
            sum(
                [
                    torch.trace(torch.mm(a, a)) * torch.trace(torch.mm(g, g))
                    for a, g in self.data.values()
                ]
            )
            ** 0.5
        )

    def compute_eigendecomposition(self, impl="eigh"):
        self.evals = dict()
        self.evecs = dict()
        if impl == "eigh":
            for layer_id in self.layer_collection.layers.keys():
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
            prod[layer_id] = (torch.mm(a, a_other), torch.mm(g, g_other))
        return PMatKFAC(
            generator=self.generator, data=prod, layer_collection=self.layer_collection
        )


class PMatEKFAC(PMatAbstract):
    """
    EKFAC representation from
    *George, Laurent et al., Fast Approximate Natural Gradient Descent
    in a Kronecker-factored Eigenbasis, NeurIPS 2018*

    """

    def __init__(
        self,
        layer_collection,
        generator,
        data=None,
        examples=None,
        eigendecomposition=None,
        **kwargs,
    ):
        self._check_data_examples(data, examples)

        self.layer_collection = layer_collection
        self.generator = generator

        if eigendecomposition is None:
            eigendecomposition = lambda x: torch.linalg.eigh(x)

        if data is None:
            evecs = dict()
            diags = dict()

            kfac_blocks = generator.get_kfac_blocks(
                examples, layer_collection=layer_collection
            )
            for layer_id, layer in self.layer_collection.layers.items():
                a, g = kfac_blocks[layer_id]

                evals_a, evecs_a = eigendecomposition(a)
                evals_g, evecs_g = eigendecomposition(g)

                evecs[layer_id] = (evecs_a, evecs_g)
                if layer.transposed:
                    diags[layer_id] = evals_a[:, None] * evals_g[None, :]
                else:
                    diags[layer_id] = evals_g[:, None] * evals_a[None, :]
                del a, g, kfac_blocks[layer_id]
            self.data = (evecs, diags)
            self._kfac_coefficients = True
        else:
            self.data = data
            self._kfac_coefficients = False

    def get_device(self):
        return next(iter(self.data[1].values())).device

    def to_torch(self, split_weight_bias=True):
        """
        - split_weight_bias (bool): if True then the parameters are ordered in
        the same way as in the dense or blockdiag representation, but it
        involves more operations. Otherwise the coefficients corresponding
        to the bias are mixed between coefficients of the weight matrix
        """
        _, diags = self.data
        s = self.layer_collection.numel()
        dtype = next(iter(diags.values())).dtype

        M = torch.zeros(
            (s, s),
            device=self.get_device(),
            dtype=dtype,
        )
        for layer_id, layer in self.layer_collection.layers.items():
            diag = diags[layer_id]
            start = self.layer_collection.p_pos[layer_id]
            sAG = diag.numel()
            M[start : start + sAG, start : start + sAG].add_(
                self.get_block_torch(
                    layer_id, layer, split_weight_bias=split_weight_bias
                )
            )
        return M

    def get_block_torch(self, layer_id, layer, split_weight_bias=True):
        KFE = self.get_KFE(layer_id, layer, split_weight_bias=split_weight_bias)
        diag = self.data[1][layer_id]

        return torch.mm(KFE, torch.mm(torch.diag(diag.view(-1)), KFE.t()))

    def get_KFE(self, layer_id, layer, split_weight_bias=True):
        """
        Returns a dict index by layers, of dense eigenvectors constructed from
        Kronecker-factored eigenvectors

        - split_weight_bias (bool): if True then the parameters are ordered in
        the same way as in the dense or blockdiag representation, but it
        involves more operations. Otherwise the coefficients corresponding
        to the bias are mixed between coefficients of the weight matrix
        """
        evecs, _ = self.data
        evecs_a, evecs_g = evecs[layer_id]
        if split_weight_bias and layer.has_bias():
            return torch.cat(
                [
                    kronecker(evecs_g, evecs_a[:-1, :], transpose=layer.transposed),
                    kronecker(evecs_g, evecs_a[-1:, :], transpose=layer.transposed),
                ],
                dim=0,
            )
        else:
            return kronecker(evecs_g, evecs_a, transpose=layer.transposed)

    def update_diag(self, examples):
        """
        Will update the diagonal in the KFE (aka the approximate eigenvalues)
        using current values of the model's parameters
        """
        self.data = (
            self.data[0],
            self.generator.get_kfe_diag(
                self.data[0], examples, layer_collection=self.layer_collection
            ),
        )
        self._kfac_coefficients = False

    def mv(self, vs):
        self._check_diag_updated()
        vs_dict = vs.to_dict()
        out_dict = dict()
        evecs, diags = self.data

        lc_merged = self.layer_collection.merge(vs.layer_collection)
        for layer_id, layer in lc_merged.layers.items():
            diag = diags[layer_id]
            evecs_a, evecs_g = evecs[layer_id]
            if layer.transposed:
                evecs_a, evecs_g = evecs_g, evecs_a
            vw = vs_dict[layer_id][0]
            sw = vw.size()
            v = vw.view(sw[0], -1)
            if layer.has_bias():
                v = torch.cat([v, vs_dict[layer_id][1].unsqueeze(1)], dim=1)
            v_kfe = torch.mm(torch.mm(evecs_g.t(), v), evecs_a)
            mv_kfe = v_kfe * diag.view(*v_kfe.size())
            mv = torch.mm(torch.mm(evecs_g, mv_kfe), evecs_a.t())
            if layer.has_bias():
                mv_tuple = (mv[:, :-1].contiguous().view(*sw), mv[:, -1].contiguous())
            else:
                mv_tuple = (mv.view(*sw),)
            out_dict[layer_id] = mv_tuple
        return PVector(layer_collection=lc_merged, dict_repr=out_dict)

    def vTMv(self, vector):
        self._check_diag_updated()
        vector_dict = vector.to_dict()
        evecs, diags = self.data
        norm2 = 0
        lc_merged = self.layer_collection.merge(vector.layer_collection)
        for layer_id, layer in lc_merged.layers.items():
            evecs_a, evecs_g = evecs[layer_id]
            if layer.transposed:
                evecs_a, evecs_g = evecs_g, evecs_a
            diag = diags[layer_id]
            v = vector_dict[layer_id][0].view(vector_dict[layer_id][0].size(0), -1)
            if len(vector_dict[layer_id]) > 1:
                v = torch.cat([v, vector_dict[layer_id][1].unsqueeze(1)], dim=1)

            v_kfe = torch.mm(torch.mm(evecs_g.t(), v), evecs_a)
            norm2 += torch.dot(v_kfe.view(-1) ** 2, diag.view(-1))
        return norm2

    def trace(self):
        self._check_diag_updated()
        return sum([d.sum() for d in self.data[1].values()])

    def frobenius_norm(self):
        self._check_diag_updated()
        return sum([(d**2).sum() for d in self.data[1].values()]) ** 0.5

    def get_diag(self, v):
        self._check_diag_updated()
        raise NotImplementedError

    def inverse(self, regul=1e-8):
        self._check_diag_updated()
        return self.pow(-1, regul=regul)

    def pow(self, pow, regul=1e-8):
        self._check_diag_updated()
        evecs, diags = self.data
        inv_diags = {i: (d + regul) ** pow for i, d in diags.items()}
        return PMatEKFAC(
            generator=self.generator,
            data=(evecs, inv_diags),
            layer_collection=self.layer_collection,
        )

    def __pow__(self, pow):
        self._check_diag_updated()
        return self.pow(pow)

    def solvePVec(self, x, regul=1e-8, solve="default"):
        self._check_diag_updated()
        if solve != "default":
            raise NotImplementedError

        vs_dict = x.to_dict()
        out_dict = dict()
        evecs, diags = self.data
        lc_merged = self.layer_collection.merge(x.layer_collection)
        for l_id, l in lc_merged.layers.items():
            diag = diags[l_id]
            evecs_a, evecs_g = evecs[l_id]
            if l.transposed:
                evecs_a, evecs_g = evecs_g, evecs_a
            vw = vs_dict[l_id][0]
            sw = vw.size()
            v = vw.view(sw[0], -1)
            if l.has_bias():
                v = torch.cat([v, vs_dict[l_id][1].unsqueeze(1)], dim=1)
            v_kfe = torch.mm(torch.mm(evecs_g.t(), v), evecs_a)
            inv_kfe = v_kfe / (diag.view(*v_kfe.size()) + regul)
            inv = torch.mm(torch.mm(evecs_g, inv_kfe), evecs_a.t())
            if l.has_bias():
                inv_tuple = (
                    inv[:, :-1].contiguous().view(*sw),
                    inv[:, -1].contiguous(),
                )
            else:
                inv_tuple = (inv.view(*sw),)
            out_dict[l_id] = inv_tuple
        return PVector(layer_collection=lc_merged, dict_repr=out_dict)

    def solvePFMap(self, x, regul=1e-8, solve="default"):
        self._check_diag_updated()
        if solve != "default":
            raise NotImplementedError

        out_dict = OrderedDict()
        evecs, diags = self.data
        lc_merged = self.layer_collection.merge(x.layer_collection)
        for l_id, layer, vals in x.iter_by_layer():
            if l_id not in lc_merged.layers:
                continue

            diag = diags[l_id]
            evecs_a, evecs_g = evecs[l_id]
            if layer.transposed:
                evecs_a, evecs_g = evecs_g, evecs_a
            vw = vals[0]
            sw = vw.size()
            v = vw.view(sw[0], sw[1], sw[2], -1)
            if len(vals) > 1:
                vb = vals[1]
                v = torch.cat([v, vb.unsqueeze(3)], dim=3)

            v_kfe = torch.einsum("ijkl,ka,lb->ijab", v, evecs_g, evecs_a)

            sv_kfe = v_kfe.size()
            inv_kfe = v_kfe / (diag.reshape(1, sv_kfe[2], sv_kfe[3]) + regul)

            inv = torch.einsum("ijkl,ak,bl->ijab", inv_kfe, evecs_g, evecs_a)

            if len(vals) > 1:
                inv_tuple = (
                    inv[:, :, :, :-1].contiguous().view(*sw),
                    inv[:, :, :, -1].contiguous(),
                )
            else:
                inv_tuple = (inv.view(*sw),)
            out_dict[l_id] = inv_tuple

        return PFMapDense.from_dict(
            generator=self.generator,
            data_dict=out_dict,
            layer_collection=lc_merged,
        )

    def __rmul__(self, x):
        evecs, diags = self.data
        diags = {l_id: x * d for l_id, d in diags.items()}
        return PMatEKFAC(
            generator=self.generator,
            data=(evecs, diags),
            layer_collection=self.layer_collection,
        )

    def _check_diag_updated(self):
        if self._kfac_coefficients:
            warnings.warn(
                UserWarning(
                    """It is required that you call .update_diag to obtain
                          the true EKFAC matrix, otherwise the representation is equivalent
                          to using KFAC"""
                )
            )


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

    def __init__(self, layer_collection, generator, data=None, examples=None, **kwargs):
        self.generator = generator

        self.layer_collection = layer_collection
        assert data is None

        self.examples = examples

    def mv(self, v):
        return self.generator.implicit_mv(
            v, self.examples, layer_collection=self.layer_collection
        )

    def vTMv(self, v):
        return self.generator.implicit_vTMv(
            v, self.examples, layer_collection=self.layer_collection
        )

    def trace(self):
        return self.generator.implicit_trace(
            self.examples, layer_collection=self.layer_collection
        )

    def frobenius_norm(self):
        raise NotImplementedError

    def to_torch(self):
        raise NotImplementedError

    def solvePVec(self, *args):
        raise NotImplementedError

    def get_diag(self):
        raise NotImplementedError

    def get_device(self):
        return "none"


class PMatLowRank(PMatAbstract):
    def __init__(self, layer_collection, generator, data=None, examples=None, **kwargs):
        self._check_data_examples(data, examples)

        self.layer_collection = layer_collection
        self.generator = generator
        if data is not None:
            self.data = data
        else:
            self.data = generator.get_jacobian(
                examples, layer_collection=layer_collection
            )
            self.data /= self.data.size(1) ** 0.5

    def vTMv(self, v):
        data_mat = self.data.view(-1, self.data.size(2))
        Av = torch.mv(data_mat, v.to_torch())
        return torch.dot(Av, Av)

    def to_torch(self):
        # you probably don't want to do that: you are
        # loosing the benefit of having a low rank representation
        # of your matrix but instead compute the potentially
        # much larger dense matrix
        return torch.mm(
            self.data.view(-1, self.data.size(2)).t(),
            self.data.view(-1, self.data.size(2)),
        )

    def mv(self, v):
        data_mat = self.data.view(-1, self.data.size(2))
        v_flat = torch.mv(data_mat.t(), torch.mv(data_mat, v.to_torch()))
        return PVector(v.layer_collection, vector_repr=v_flat)

    def compute_eigendecomposition(self, impl="svd"):
        data_mat = self.data.view(-1, self.data.size(2))
        if impl == "svd":
            _, sqrt_evals, self.evecs = torch.svd(data_mat, some=True)
            self.evals = sqrt_evals**2
        else:
            raise NotImplementedError

    def get_eigendecomposition(self):
        return self.evals, self.evecs

    def trace(self):
        A = torch.mm(
            self.data.view(-1, self.data.size(2)),
            self.data.view(-1, self.data.size(2)).t(),
        )
        return torch.trace(A)

    def frobenius_norm(self):
        A = torch.mm(
            self.data.view(-1, self.data.size(2)),
            self.data.view(-1, self.data.size(2)).t(),
        )
        return torch.norm(A)

    def solvePVec(self, x, regul=1e-8, solve="svd"):
        if solve not in ["svd", "default"]:
            raise NotImplementedError

        u, s, v = torch.svd(self.data.view(-1, self.data.size(2)))
        d = torch.mv(v, torch.mv(v.t(), x.to_torch()) / (s**2 + regul))
        return PVector(x.layer_collection, vector_repr=d)

    def get_diag(self):
        return (self.data**2).sum(dim=(0, 1))

    def __rmul__(self, x):
        return PMatLowRank(
            generator=self.generator,
            data=x**0.5 * self.data,
            layer_collection=self.layer_collection,
        )

    def get_device(self):
        return self.data.device


class PMatQuasiDiag(PMatAbstract):
    """
    Quasidiagonal approximation as decribed in Ollivier,
    Riemannian metrics for neural networks I: feedforward networks,
    Information and Inference: A Journal of the IMA, 2015
    """

    def __init__(self, layer_collection, generator, data=None, examples=None, **kwargs):
        self._check_data_examples(data, examples)

        self.layer_collection = layer_collection
        self.generator = generator
        if data is not None:
            self.data = data
        else:
            self.data = generator.get_covariance_quasidiag(
                examples, layer_collection=layer_collection
            )

    def get_device(self):
        return next(iter(self.data.values()))[0].device

    def to_torch(self):
        s = self.layer_collection.numel()
        M = torch.zeros((s, s), device=self.get_device())
        for layer_id in self.layer_collection.layers.keys():
            diag, cross = self.data[layer_id]
            block_s = diag.size(0)
            block = torch.diag(diag)
            if cross is not None:
                out_s = cross.size(0)
                in_s = cross.numel() // out_s

                block_bias = torch.cat(
                    (
                        cross.view(cross.size(0), -1).t().reshape(-1, 1),
                        torch.zeros((out_s * in_s, out_s), device=self.get_device()),
                    ),
                    dim=1,
                )
                block_bias = (
                    block_bias.view(in_s, out_s + 1, out_s)
                    .transpose(0, 1)
                    .reshape(-1, out_s)[: in_s * out_s, :]
                )

                block[: in_s * out_s, in_s * out_s :].copy_(block_bias)
                block[in_s * out_s :, : in_s * out_s].copy_(block_bias.t())
            start = self.layer_collection.p_pos[layer_id]
            M[start : start + block_s, start : start + block_s].add_(block)
        return M

    def frobenius_norm(self):
        norm2 = 0
        for layer_id in self.layer_collection.layers.keys():
            diag, cross = self.data[layer_id]
            norm2 += torch.dot(diag, diag)
            if cross is not None:
                norm2 += 2 * torch.dot(cross.view(-1), cross.view(-1))

        return norm2**0.5

    def get_diag(self):
        return torch.cat(
            [self.data[l_id][0] for l_id in self.layer_collection.layers.keys()]
        )

    def trace(self):
        return sum(
            [self.data[l_id][0].sum() for l_id in self.layer_collection.layers.keys()]
        )

    def vTMv(self, vs):
        vs_dict = vs.to_dict()
        out = 0
        for layer_id, layer in self.layer_collection.layers.items():
            diag, cross = self.data[layer_id]
            v_weight = vs_dict[layer_id][0]
            if layer.bias is not None:
                v_bias = vs_dict[layer_id][1]
            mv_bias = None
            mv_weight = diag[: layer.weight.numel()] * v_weight.view(-1)
            if layer.bias is not None:
                mv_bias = diag[layer.weight.numel() :] * v_bias.view(-1)
                mv_bias += (cross * v_weight).view(v_bias.size(0), -1).sum(dim=1)
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
        vs_dict = vs.to_dict()
        out_dict = dict()
        for layer_id, layer in self.layer_collection.layers.items():
            diag, cross = self.data[layer_id]

            v_weight = vs_dict[layer_id][0]
            if layer.bias is not None:
                v_bias = vs_dict[layer_id][1]

            mv_bias = None
            mv_weight = diag[: layer.weight.numel()].view(*v_weight.size()) * v_weight
            if layer.bias is not None:
                mv_bias = diag[layer.weight.numel() :] * v_bias.view(-1)
                mv_bias += (cross * v_weight).view(v_bias.size(0), -1).sum(dim=1)
                if len(cross.size()) == 2:
                    mv_weight += cross * v_bias.view(-1, 1)
                elif len(cross.size()) == 4:
                    mv_weight += cross * v_bias.view(-1, 1, 1, 1)
                else:
                    raise NotImplementedError

            out_dict[layer_id] = (mv_weight, mv_bias)
        return PVector(layer_collection=vs.layer_collection, dict_repr=out_dict)

    def solvePVec(self, x, regul=1e-8, solve="default"):
        if solve != "default":
            raise NotImplementedError

        vs_dict = x.to_dict()
        out_dict = dict()
        for layer_id, layer in self.layer_collection.layers.items():
            diag, cross = self.data[layer_id]

            v_weight = vs_dict[layer_id][0]
            # keep original size
            s_w = v_weight.size()
            v_weight = v_weight.view(s_w[0], -1)

            d_weight = diag[: layer.weight.numel()].view(s_w[0], -1) + regul
            solve_b = None
            if layer.bias is None:
                solve_w = v_weight / d_weight
            else:
                v_bias = vs_dict[layer_id][1]
                d_bias = diag[layer.weight.numel() :] + regul

                cross = cross.view(s_w[0], -1)

                solve_b_denom = d_bias - bdot(cross / d_weight, cross)
                solve_b = (v_bias - bdot(cross / d_weight, v_weight)) / solve_b_denom

                solve_w = (v_weight - solve_b.unsqueeze(1) * cross) / d_weight

            out_dict[layer_id] = (solve_w.view(*s_w), solve_b)
        return PVector(layer_collection=x.layer_collection, dict_repr=out_dict)


class PMatMixed(PMatAbstract):
    def __init__(
        self,
        layer_collection,
        generator,
        default_representation,
        map_layers_to,
        data=None,
        examples=None,
        **kwargs,
    ):
        self.generator = generator

        self.layer_collection = layer_collection
        self.layer_collection_each = defaultdict(LayerCollection)
        self.layer_map = dict()

        for layer_id, layer in layer_collection.layers.items():
            layer_type = layer.__class__
            representation = default_representation
            if layer_type in map_layers_to:
                representation = map_layers_to[layer_type]
            self.layer_collection_each[representation].add_layer(layer_id, layer)
            self.layer_map[layer_id] = representation

        self.sub_pmats = {
            PMat_class: PMat_class(
                layer_collection=lc,
                generator=generator,
                data=data,
                examples=examples,
                **kwargs,
            )
            for PMat_class, lc in self.layer_collection_each.items()
        }

        # hardcoded :-(
        if PMatEKFAC in self.sub_pmats.keys():
            self.update_diag = self.sub_pmats[PMatEKFAC].update_diag

    def frobenius_norm(self):
        return (
            sum([pmat.frobenius_norm() ** 2 for pmat in self.sub_pmats.values()]) ** 0.5
        )

    def get_device(self):
        device = None
        for pmat in self.sub_pmats.values():
            if device is None:
                device = pmat.get_device()
            elif device != pmat.get_device():
                raise NotImplementedError("All blocks should be on the same device")
        return device

    def get_diag(self):
        raise NotImplementedError()

    def mv(self, v):
        out_dict = dict()
        for pmat in self.sub_pmats.values():
            out_dict |= pmat.mv(v).to_dict()
        return PVector(layer_collection=self.layer_collection, dict_repr=out_dict)

    def solvePVec(self, x, *args, **kwargs):
        out_dict = dict()
        for prepr, pmat in self.sub_pmats.items():
            if (
                "solve" in kwargs
                and type(kwargs["solve"]) is dict
                and prepr in kwargs["solve"]
            ):
                kwargs_dispatch = kwargs | {"solve": kwargs["solve"][prepr]}
            else:
                kwargs_dispatch = kwargs
            out_dict |= pmat.solvePVec(x, *args, **kwargs_dispatch).to_dict()
        return PVector(layer_collection=self.layer_collection, dict_repr=out_dict)

    def solvePFMap(self, x, *args, **kwargs):
        out_dict = dict()
        for prepr, pmat in self.sub_pmats.items():
            if (
                "solve" in kwargs
                and type(kwargs["solve"]) is dict
                and prepr in kwargs["solve"]
            ):
                kwargs_dispatch = kwargs | {"solve": kwargs["solve"][prepr]}
            else:
                kwargs_dispatch = kwargs
            out_dict |= {
                k: v
                for k, _, v in pmat.solvePFMap(
                    x, *args, **kwargs_dispatch
                ).iter_by_layer()
            }
        return PFMapDense.from_dict(
            layer_collection=self.layer_collection,
            generator=self.generator,
            data_dict=out_dict,
        )

    def trace(self):
        return sum([pmat.trace() for pmat in self.sub_pmats.values()])

    def vTMv(self, v):
        return sum([pmat.vTMv(v) for pmat in self.sub_pmats.values()])

    def to_torch(self):
        s = self.layer_collection.numel()
        M = None
        for layer_id, layer in self.layer_collection.layers.items():
            numel = layer.numel()
            start = self.layer_collection.p_pos[layer_id]
            block = self.sub_pmats[self.layer_map[layer_id]].get_block_torch(
                layer_id, layer
            )
            if M is None:
                M = torch.zeros((s, s), device=self.get_device(), dtype=block.dtype)

            M[start : start + numel, start : start + numel].add_(block)
        return M


class PMatEKFACBlockDiag(PMatMixed):
    """A mixed representation where EKFAC-table layers use EKFAC,
    and other layers use a block-diagonal matrix"""

    def __init__(self, layer_collection, generator, data=None, examples=None, **kwargs):
        super().__init__(
            layer_collection,
            generator,
            data=data,
            examples=examples,
            default_representation=PMatBlockDiag,
            map_layers_to={
                LinearLayer: PMatEKFAC,
                Conv1dLayer: PMatEKFAC,
                Conv2dLayer: PMatEKFAC,
                EmbeddingLayer: PMatEKFAC,
            },
            **kwargs,
        )


class PMatEye(PMatAbstract):
    def __init__(self, layer_collection, scaling=torch.tensor(1.0), **kwargs):
        self.layer_collection = layer_collection
        self.scaling = scaling
        self.generator = None

    def solvePVec(self, v, regul=1e-8, solve="default"):
        if solve != "default":
            raise NotImplementedError
        solution = v.to_torch() / (self.scaling + regul)
        return PVector(layer_collection=v.layer_collection, vector_repr=solution)

    def vTMv(self, v):
        v_flat = v.to_torch()
        return self.scaling * torch.dot(v_flat, v_flat)

    def frobenius_norm(self):
        return self.size(0) ** 0.5 * torch.abs(self.scaling)

    def mv(self, v):
        v_flat = v.to_torch() * self.scaling
        return PVector(v.layer_collection, vector_repr=v_flat)

    def trace(self):
        return self.scaling * self.size(0)

    def get_device(self):
        return self.scaling.device

    def get_diag(self):
        return self.scaling * torch.ones(
            self.size(0), dtype=self.scaling.dtype, device=self.scaling.device
        )

    def to_torch(self):
        return self.scaling * torch.eye(
            self.size(0), dtype=self.scaling.dtype, device=self.scaling.device
        )

    def __rmul__(self, x):
        return PMatEye(
            layer_collection=self.layer_collection,
            scaling=x * self.scaling,
        )


def bdot(A, B):
    """
    batched dot product
    """
    return torch.matmul(A.unsqueeze(1), B.unsqueeze(2)).squeeze(1).squeeze(1)
