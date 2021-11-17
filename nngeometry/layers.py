from torch import Tensor
from torch.nn import Linear, Conv2d, Module, init
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import torch

class Cosine1d(Linear):
    """Computes the cosine similarity between rows of the weight matrix
    and the incoming data
    """

    def __init__(self, in_features: int, out_features: int, eps=1e-05) -> None:
        super(Cosine1d, self).__init__(in_features=in_features,
                                     out_features=out_features,
                                     bias=False)
        self.eps = eps

    def forward(self, input: Tensor) -> Tensor:
        norm2_w = (self.weight**2).sum(dim=1, keepdim=True) + self.eps
        norm2_x = (input**2).sum(dim=1, keepdim=True) + self.eps
        return F.linear(input / torch.sqrt(norm2_x),
                        self.weight / torch.sqrt(norm2_w))


class WeightNorm1d(Linear):
    """Computes an affine mapping of the incoming data using a weight matrix
    with rows normalized with norm 1
    """

    def __init__(self, in_features: int, out_features: int, eps=1e-05) -> None:
        super(WeightNorm1d, self).__init__(in_features=in_features,
                                     out_features=out_features,
                                     bias=False)
        self.eps = eps

    def forward(self, input: Tensor) -> Tensor:
        norm2 = (self.weight**2).sum(dim=1, keepdim=True) + self.eps
        return F.linear(input,
                        self.weight / torch.sqrt(norm2))


class WeightNorm2d(Conv2d):
    """Computes a 2d convolution using a kernel weight matrix
    with rows normalized with norm 1
    """

    def __init__(self, *args, eps=1e-05, **kwargs) -> None:
        assert 'bias' not in kwargs or kwargs['bias'] is False
        super(WeightNorm2d, self).__init__(*args, bias=False, **kwargs)
        self.eps = eps

    def forward(self, input: Tensor) -> Tensor:
        norm2 = (self.weight**2).sum(dim=(1, 2, 3), keepdim=True) + self.eps
        return self._conv_forward(input, self.weight / torch.sqrt(norm2),
                                  None)


class Affine1d(Module):
    """Computes the transformation out = weight * input + bias
    where * is the elementwise multiplication. This is similar to the
    scaling and translation given by parameters gamma and beta in batch norm

    """
    def __init__(self, num_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Affine1d, self).__init__()
        self.num_features = num_features
        self.weight = Parameter(torch.empty(num_features, **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(num_features, **factory_kwargs))
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
        return 'num_features={}, bias={}'.format(
            self.num_features, self.bias is not None
        )