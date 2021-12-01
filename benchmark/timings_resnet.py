# %%
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
import time
import pprint


from nngeometry.layercollection import LayerCollection
from nngeometry.metrics import FIM_MonteCarlo
from nngeometry.object.vector import random_pvector
from nngeometry.generator import jacobian as nnj

from nngeometry.object import PMatDiag, PMatKFAC, PMatEKFAC, PMatQuasiDiag, PMatImplicit


# # ResNet50 on CIFAR10

# %%
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

trainset = datasets.CIFAR10(root='/tmp/data', train=True,
                            download=True, transform=transform)
trainset = Subset(trainset, range(100))
trainloader = DataLoader(trainset, batch_size=50,
                         shuffle=False, num_workers=1)

# %%
from resnet import ResNet50
resnet = ResNet50().cuda()

layer_collection = LayerCollection.from_model(resnet)
v = random_pvector(LayerCollection.from_model(resnet), device='cuda')

print(f'{layer_collection.numel()} parameters')

# %%
# compute timings and display FIMs

def perform_timing():
    timings = dict()

    for repr in [PMatImplicit, PMatDiag, PMatEKFAC, PMatKFAC, PMatQuasiDiag]:
        
        print('Timing representation:')
        pprint.pprint(repr)
        
        timings[repr] = dict()
        
        time_start = time.time()
        F = FIM_MonteCarlo(model=resnet,
                        loader=trainloader,
                        representation=repr,
                        device='cuda')
        time_end = time.time()
        timings[repr]['init'] = time_end - time_start
        
        if repr == PMatEKFAC:
            time_start = time.time()
            F.update_diag(examples=trainloader)
            time_end = time.time()
            timings[repr]['update_diag'] = time_end - time_start
            
        time_start = time.time()
        F.mv(v)
        time_end = time.time()
        timings[repr]['Mv'] = time_end - time_start
        
        time_start = time.time()
        F.vTMv(v)
        time_end = time.time()
        timings[repr]['vTMv'] = time_end - time_start
        
        time_start = time.time()
        F.trace()
        time_end = time.time()
        timings[repr]['tr'] = time_end - time_start
        
        try:
            time_start = time.time()
            F.frobenius_norm()
            time_end = time.time()
            timings[repr]['frob'] = time_end - time_start
        except NotImplementedError:
            pass
        
        try:
            time_start = time.time()
            F.solve(v)
            time_end = time.time()
            timings[repr]['solve'] = time_end - time_start
        except:
            pass
        
        del F

    pprint.pprint(timings)

# %%

with nnj.use_unfold_impl_for_convs():
    perform_timing()

with nnj.use_conv_impl_for_convs():
    perform_timing()