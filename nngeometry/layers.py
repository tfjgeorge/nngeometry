from torch import Tensor
from torch.nn import Linear
from torch.nn import functional as F
import torch

class Cosine(Linear):
    """Computes the cosine similarity between rows of the weight matrix
    and the incoming data
    """

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input / torch.norm(input, dim=1, keepdim=True),
                        self.weight / torch.norm(self.weight, dim=1, keepdim=True),
                        self.bias)