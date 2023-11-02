import torch

from nngeometry.maths import kronecker


def test_kronecker():
    eps = 1e-5

    A = torch.tensor([[1.0, 0.0]])
    B = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    AkronB = torch.tensor([[1.0, 2.0, 0.0, 0.0], [3.0, 4.0, 0.0, 0.0]])
    assert torch.norm(kronecker(A, B) - AkronB) < eps

    A = torch.tensor([[1.0], [2.0]])
    AkronB = torch.tensor([[1.0, 2.0], [3.0, 4.0], [2.0, 4.0], [6.0, 8.0]])
    assert torch.norm(kronecker(A, B) - AkronB) < eps
