import torch
import torch.nn.functional as F


def get_n_parameters(model):
    return sum([p.numel() for p in model.parameters()])


def per_example_grad_conv(mod, x, gy):
    ks = (mod.weight.size(2), mod.weight.size(3))
    gy_s = gy.size()
    bs = gy_s[0]
    x_unfold = F.unfold(x, kernel_size=ks, stride=mod.stride,
                        padding=mod.padding, dilation=mod.dilation)
    x_unfold_s = x_unfold.size()
    return torch.bmm(gy.view(bs, gy_s[1], -1),
                     x_unfold.view(bs, x_unfold_s[1], -1).permute(0, 2, 1))


def display_correl(M, axis):

    M = M.get_dense_tensor()
    diag = torch.diag(M)
    dM = (diag + diag.mean() / 100) **.5
    correl = torch.abs(M) / dM.unsqueeze(0) / dM.unsqueeze(1)

    axis.imshow(correl.cpu())