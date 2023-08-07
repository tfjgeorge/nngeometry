#%% I'm trying to use the latest git release of NNGeometry's FIM to find the Fisher metric of my trivial model. As a simple example  I create a model which has a single Linear layer, a single training sample, and solves the matrix equation Ax=b, where A is a 3x3 matrix, whilst x, b are 3x1 col. vectors.

# Here's my code (it's not meant for anything functional -- it's just to see how these things work):

# ```
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
# ```

# Now I create a dataloader with a single batch containing the single training sample:

# ```
from torch.utils.data import DataLoader, Dataset

class TrivialDataset(Dataset):
    def __init__(self):
        self.data = torch.arange(1, 10, dtype=torch.float32).view(1,9)
    def __getitem__(self, index):
        return (self.data[index], )

    def __len__(self):
        return len(self.data)

batch_size = 1
dataset = TrivialDataset()
loader = DataLoader(dataset, batch_size=batch_size)
# ```

# Attempting to compute the Fisher metrics gives a runtime error due to the differentiated tensors not being used.
# `

# ```
from nngeometry.metrics import FIM
from nngeometry.object import PMatDense, PMatBlockDiag
# check dimensions
print(model)
fisher_metric = FIM(model, loader, n_output=3, variant='regression', representation=PMatDense, device='cpu')
# ```

# RuntimeError: One of the differentiated Tensors appears to not have been used in the graph. Set allow_unused=True if this is the desired behavior.`

# I'm at an utter loss as to why this is happening. Is this a bug in NNGeometry (unlikely) or am I doing something extremely stupid (increasingly likely)? Thanks!

# %%
