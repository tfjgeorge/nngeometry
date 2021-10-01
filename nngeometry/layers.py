from torch import Tensor
from torch.nn import Linear, Module
from torch.nn import functional as F
from torch.nn.parameter import Parameter
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


class Affine1d(Module):
    def __init__(self, n_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Affine1d, self).__init__()
        self.n_features = n_features
        self.weight = Parameter(torch.ones(n_features, **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.zeros(n_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, input: Tensor) -> Tensor:
        if self.bias is not None:
            return input * self.weight.unsqueeze(0) + self.bias
        else:
            return input * self.weight.unsqueeze(0)

    def extra_repr(self) -> str:
        return 'n_features={}, bias={}'.format(
            self.n_features, self.bias is not None
        )