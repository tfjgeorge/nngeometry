import torch
import pytest


@pytest.fixture(scope="session", autouse=True)
def default_to_float64_on_cpu(request):
    if not torch.cuda.is_available(): 
        torch.set_default_dtype(torch.float64)
        torch.set_default_tensor_type(torch.DoubleTensor)