import torch
from tasks import device


def check_ratio(vref, v2, eps=1e-3):
    if vref == 0:
        ratio = v2
    else:
        ratio = (v2 - vref) / vref
    assert ratio < eps and -ratio < eps


def check_tensors(tref, t2, eps=1e-3, only_print_diff=False):
    if torch.norm(tref) == 0:
        relative_diff = torch.norm(t2 - tref)
    else:
        relative_diff = torch.norm(t2 - tref) / torch.norm(tref)
    if only_print_diff:
        print(relative_diff)
    else:
        assert relative_diff < eps
    return relative_diff


def check_angle(v1, v2, eps=1e-3):
    cos_angle = torch.dot(v1.view(-1), v2.view(-1)) / torch.norm(v1) / torch.norm(v2)
    assert cos_angle < 1 + eps and cos_angle > 1 - eps


def angle(v1, v2):
    v1_flat = v1.to_torch()
    v2_flat = v2.to_torch()
    return torch.dot(v1_flat, v2_flat) / (torch.norm(v1_flat) * torch.norm(v2_flat))


def update_model(parameters, dw):
    i = 0
    for p in parameters:
        j = i + p.numel()
        p.data += dw[i:j].view(*p.size())
        i = j


def get_output_vector(loader, function):
    with torch.no_grad():
        outputs = []
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs.append(function(inputs, targets))
        return torch.cat(outputs)
