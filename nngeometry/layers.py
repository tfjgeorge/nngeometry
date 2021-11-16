from torch import Tensor
from torch.nn import Linear, Conv2d, Module, init
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


class WeightNorm2d(Conv2d):
    """Computes an 2d convolution using a kernel weight matrix
    with rows normalized with norm 1
    """

    def __init__(self, *args, **kwargs) -> None:
        assert 'bias' not in kwargs or kwargs['bias'] is False
        super(WeightNorm2d, self).__init__(*args, **kwargs)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input,
                                  self.weight / torch.norm(self.weight, dim=(1, 2, 3),
                                                           keepdim=True))


class Affine1d(Module):
    """Computes the transformation out = weight * input + bias
    where * is the elementwise multiplication. This is similar to the
    scaling and translation given by parameters gamma and beta in batch norm

    """
    def __init__(self, n_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Affine1d, self).__init__()
        self.n_features = n_features
        self.weight = Parameter(torch.empty(n_features, **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(n_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.ones_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        if self.bias is not None:
            return input * self.weight.unsqueeze(0) + self.bias
        else:
            return input * self.weight.unsqueeze(0)

    def extra_repr(self) -> str:
        return 'n_features={}, bias={}'.format(
            self.n_features, self.bias is not None
        )