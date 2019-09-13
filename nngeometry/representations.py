import torch

class DenseMatrix:
    def __init__(self, generator):
        self.generator = generator
        self.data = generator.get_matrix()
        # self.evals, self.evecs = self.get_eigendecomposition()

    def mv(self, v):
        return torch.mv(self.data, v)

    def m_norm(self, v):
        return torch.dot(v, torch.mv(self.data, v)) ** .5

    def project_to_diag(self, v):
        # TODO test
        return torch.mv(self.evecs.t(), v)

    def project_from_diag(self, v):
        # TODO test
        return torch.mv(self.evecs, v)

    def size(self, *args):
        return self.data.size(*args)

    def get_eigendecomposition(self):
        # TODO test
        return torch.symeig(self.data, eigenvectors=True)

    def trace(self):
        return self.data.trace()

    def get_matrix(self):
        return self.data

class ImplicitMatrix:
    def __init__(self, generator):
        self.generator = generator

    def mv(self, v):
        return self.generator.implicit_mv(v)

    def m_norm(self, v):
        return self.generator.implicit_m_norm(v)