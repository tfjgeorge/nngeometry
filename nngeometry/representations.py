import torch

class DenseMatrix:
    def __init__(self, generator):
        self.generator = generator
        self.matrix = generator.get_matrix()

    def mv(self, v):
        return torch.mv(self.matrix, v)

    def size(self, *args):
        return self.matrix.size(*args)

    def get_eigendecomposition(self):
        pass