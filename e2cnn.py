import torch
import torchvision 
import torchvision.transforms as transforms
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F
import autograd.numpy as anp
import torch.optim as optim
from MiraBest import MiraBest
from nngeometry.metrics import FIM

transforms = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])]
)

