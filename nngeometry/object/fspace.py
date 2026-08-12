import warnings
from abc import ABC, abstractmethod
from functools import cache

import torch

from .vector import FVector, PVector


class FMatAbstract(ABC):
    @abstractmethod
    def __init__(self, layer_collection, generator, data=None, examples=None):
        pass

    def __matmul__(self, other):
        if isinstance(other, FVector):
            return self.mv(other)
        elif isinstance(other, FMatAbstract):
            return self.mm(other)
        else:
            return NotImplemented

    # assumes symetric by default
    def adjoint(self):
        return self

    def __rmatmul__(self, other):
        return self.adjoint() @ other

    @abstractmethod
    def solveFVec(self, x, regul, solve, **kwargs):
        pass

    @abstractmethod
    def solveFMat(self, x, regul, solve, **kwargs):
        pass

    def solve(self, x, regul=1e-8, solve="default", **kwargs):
        """
        Solves Kx = b in x

        :param regul: regularization, depending of the type of solve (e.g. Tikhonov damping,
            or high-pass filter)
        :type regul: float
        :param b: b
        :type b: FVector or FMat
        :param solve: solve implementation, this is dependent on the FMat representation
        """
        if isinstance(x, FVector):
            return self.solveFVec(x, regul=regul, solve=solve, **kwargs)
        elif isinstance(x, FMatDense):
            return self.solveFMat(x, regul=regul, solve=solve, **kwargs)
        else:
            raise NotImplementedError("`x` should be an instance of FVector or FMat")


class FMatDense(FMatAbstract):
    def __init__(self, layer_collection, generator, data=None, examples=None):
        self.layer_collection = layer_collection
        self.generator = generator
        if data is not None:
            self.data = data
        else:
            self.data = generator.get_gram_matrix(examples, layer_collection)

    def mv(self, v):
        s = self.data.size()
        M = self.data.view(s[0] * s[1], s[2] * s[3])
        v_flat = v.to_torch().view(-1)
        return FVector(vector_repr=torch.mv(M, v_flat).view(s[0], s[1]))

    def mm(self, fmat):
        sM = self.data.size()
        M = self.data.view(-1, sM[2] * sM[3])
        sN = fmat.data.size()
        N = fmat.data.view(sN[0] * sN[1], -1)
        return FMatDense(
            self.layer_collection,
            self.generator,
            data=torch.mm(M, N).view(sM[0], sM[1], sN[2], sN[3]),
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

    def size(self, *args):
        return self.data.size(*args)

    def to_torch(self):
        return self.data

    def __add__(self, other):
        return FMatDense(
            layer_collection=self.layer_collection,
            generator=self.generator,
            data=self.data + other.data,
        )

    def __sub__(self, other):
        return FMatDense(
            layer_collection=self.layer_collection,
            generator=self.generator,
            data=self.data - other.data,
        )

    def __rmul__(self, other):
        return FMatDense(
            layer_collection=self.layer_collection,
            generator=self.generator,
            data=other * self.data,
        )

    def adjoint(self):
        return FMatDense(
            self.layer_collection,
            self.generator,
            data=self.data.permute(2, 3, 0, 1),
        )

    def vTMv(self, v):
        return v @ self.mv(v)

    def mTMm(self, fmat):
        return fmat.adjoint() @ self.mm(fmat)

    def compute_eigendecomposition(self, impl="eigh"):
        s = self.data.size()
        M = self.data.view(s[0] * s[1], s[2] * s[3])
        if impl == "eigh":
            self.evals, self.evecs = torch.linalg.eigh(M)
        elif impl == "svd":
            _, S, Vh = torch.linalg.svd(M, full_matrices=True)
            self.evals, self.evecs = S.flip(0), Vh.flip(0).t()
        else:
            raise NotImplementedError

    def get_eigendecomposition(self):
        return self.evals, self.evecs

    def trace(self):
        s = self.data.size()
        return torch.trace(self.data.view(s[0] * s[1], s[2] * s[3]))

    def __pow__(self, other):
        s = self.data.size()
        return FMatDense(
            layer_collection=self.layer_collection,
            generator=self.generator,
            data=torch.linalg.matrix_power(
                self.data.view(s[0] * s[1], s[2] * s[3]), other
            ).view(*s),
        )

    def inv(self, regul=1e-8):
        s = self.data.size()
        return FMatDense(
            layer_collection=self.layer_collection,
            generator=self.generator,
            data=torch.linalg.inv(
                self.data.view(s[0] * s[1], s[2] * s[3])
                + (regul * s[1])
                * torch.eye(
                    s[0] * s[1],
                    s[2] * s[3],
                    dtype=self.data.dtype,
                    device=self.data.device,
                )
            ).view(*s),
        )

    @cache
    def _cholesky(self, regul=1e-8):
        s = self.data.size()
        return torch.linalg.cholesky(
            self.data.view(s[0] * s[1], s[2] * s[3])
            + (regul * s[1])
            * torch.eye(
                s[0] * s[1], s[2] * s[3], device=self.data.device, dtype=self.data.dtype
            )
        )

    def solveFVec(self, v, regul=1e-8, solve="default"):
        s = self.data.size()
        v_flat = v.to_torch().view(-1, 1)
        if solve in ["default", "solve"]:
            solution = torch.cholesky_solve(v_flat, self._cholesky(regul))
        else:
            raise NotImplementedError

        return FVector(vector_repr=solution.view(s[0], s[1]))

    def solveFMat(self, fmat, regul=1e-8, solve="default"):
        s = self.data.size()
        sK = fmat.size()
        K = fmat.to_torch().view(sK[0] * sK[1], -1)
        if solve in ["default", "solve"]:
            solution = torch.cholesky_solve(K, self._cholesky(regul))
        else:
            raise NotImplementedError

        return FMatDense(
            layer_collection=self.layer_collection,
            generator=self.generator,
            data=solution.view(s[0], s[1], sK[2], sK[3]),
        )
