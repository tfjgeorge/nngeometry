# %%

import torch

import torch.nn as nn
import torch.optim as optim

class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        out = self.linear(x)
        return out

model = nn.Linear(9, 3, bias=False)

# Define the training data
A = nn.Parameter(torch.tensor([[1., 2., 3.],
                               [4., 5., 6.],
                               [7., 8., 9.]]))

b = nn.Parameter(torch.tensor([[52.],
                              [124.],
                              [196.]]))

# Define the model and the optimizer
# model = Net(input_dim=9, output_dim=3)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model
for epoch in range(2000):
    optimizer.zero_grad()
    y_pred = model(A.view(9))
    print(A@y_pred)
    loss = nn.MSELoss(reduction='sum')(A@y_pred.view(3,1), b)
    loss.backward()
    optimizer.step()
# %%
from torch.utils.data import DataLoader, Dataset

class TrivialDataset(Dataset):
    def __init__(self):
        self.data = torch.arange(1, 19, dtype=torch.float32).view(2,9)
    def __getitem__(self, index):
        return (self.data[index], )

    def __len__(self):
        return len(self.data)

batch_size = 1
dataset = TrivialDataset()
loader = DataLoader(dataset, batch_size=batch_size)
# %%

from nngeometry.metrics import FIM
from nngeometry.object import PMatDense, PMatBlockDiag
# check dimensions
print(model)
fisher_metric = FIM(model, loader, n_output=3, variant='regression', representation=PMatDense, device='cpu')
# %%
mb = next(iter(loader))
# %%
model(mb)
# %%
