import torch
import torch.nn.functional as F


def get_n_parameters(model):
    return sum([p.numel() for p in model.parameters()])


def get_individual_modules(model):
    mods = []
    sizes_mods = []
    parameters = []
    start = 0
    p_pos = dict()
    for mod in model.modules():
        mod_class = mod.__class__.__name__
        if mod_class in ['Linear', 'Conv2d']:
            mods.append(mod)
            p_pos[mod] = start
            sizes_mods.append(mod.weight.size())
            parameters.append(mod.weight)
            start += mod.weight.numel()
            if mod.bias is not None:
                sizes_mods.append(mod.bias.size())
                parameters.append(mod.bias)
                start += mod.bias.numel()

    # check order of flattening
    sizes_flat = [p.size() for p in model.parameters() if p.requires_grad]
    assert sizes_mods == sizes_flat
    # check that all parameters were added
    # will fail if using exotic layers such as BatchNorm
    assert len(set(parameters) - set(model.parameters())) == 0
    return mods, p_pos


def per_example_grad_conv(mod, x, gy):
    ks = (mod.weight.size(2), mod.weight.size(3))
    gy_s = gy.size()
    bs = gy_s[0]
    x_unfold = F.unfold(x, kernel_size=ks, stride=mod.stride,
                        padding=mod.padding, dilation=mod.dilation)
    x_unfold_s = x_unfold.size()
    return torch.bmm(gy.view(bs, gy_s[1], -1),
                     x_unfold.view(bs, x_unfold_s[1], -1).permute(0, 2, 1))
