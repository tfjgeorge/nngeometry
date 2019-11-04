import torch
from nngeometry.maths import kronecker

def test_kronecker():
    eps = 1e-5

    A = torch.tensor([[1., 0.]])
    B = torch.tensor([[1., 2.],
                      [3., 4.]])
    AkronB = torch.tensor([[1., 2., 0., 0.],
                           [3., 4., 0., 0.]])
    assert torch.norm(kronecker(A, B) - AkronB) < eps

    A = torch.tensor([[1.], [2.]])
    AkronB = torch.tensor([[1., 2.],
                           [3., 4.],
                           [2., 4.],
                           [6., 8.]])
    assert torch.norm(kronecker(A, B) - AkronB) < eps
