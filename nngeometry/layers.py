from torch import Tensor
from torch.nn import Linear, Module
from torch.nn import functional as F
import torch

class Cosine1d(Linear):
    """Computes the cosine similarity between rows of the weight matrix
    and the incoming data
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super(Cosine1d, self).__init__(in_features=in_features,
                                     out_features=out_features,
                                     bias=False)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input / torch.norm(input, dim=1, keepdim=True),
                        self.weight / torch.norm(self.weight, dim=1, keepdim=True))


class WeightNorm1d(Linear):
    """Computes an affine mapping of the incoming data using a weight matrix
    with rows normalized with norm 1
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super(WeightNorm1d, self).__init__(in_features=in_features,
                                     out_features=out_features,
                                     bias=False)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input,
                        self.weight / torch.norm(self.weight, dim=1, keepdim=True))

# class Affine(Module):
