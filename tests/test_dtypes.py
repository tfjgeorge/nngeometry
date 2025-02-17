import pytest
import torch as th

from nngeometry.metrics import FIM
from nngeometry.object import PMatDense, PMatDiag, PMatEKFAC, PMatKFAC


class SimpleModel(th.nn.Module):
    def __init__(self, dtype1, dtype2):
        super().__init__()
        self.fc1 = th.nn.Linear(10, 5, bias=True, dtype=dtype1)
        self.fc2 = th.nn.Linear(5, 2, bias=True, dtype=dtype2)

    def forward(self, x):
        return th.nn.Softmax(dim=-1)(self.fc2(self.fc1(x)))


def test_same_dtype():
    model = SimpleModel(dtype1=th.float32, dtype2=th.float64)
    dataset = th.utils.data.TensorDataset(
        th.randn(100, 10, dtype=th.float64), th.randint(0, 2, (100,))
    )
    loader = th.utils.data.DataLoader(dataset, batch_size=10)

    for PMatType in [PMatDense, PMatDiag]:
        with pytest.raises(ValueError):
            FIM(model, loader, PMatType, variant="classif_logits")


def test_dtypes():
    for dtype in [th.float32, th.float64]:
        model = SimpleModel(dtype1=dtype, dtype2=dtype)
        dataset = th.utils.data.TensorDataset(
            th.randn(100, 10, dtype=dtype), th.randint(0, 2, (100,))
        )
        loader = th.utils.data.DataLoader(dataset, batch_size=10)

    for PMatType in [PMatDense, PMatDiag, PMatKFAC, PMatEKFAC]:
        FIM(model, loader, PMatType, variant="classif_logits")
