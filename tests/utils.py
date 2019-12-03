import torch


def check_ratio(vref, v2, eps=1e-3):
    ratio = v2 / vref
    assert ratio < 1 + eps and ratio > 1 - eps


def check_tensors(tref, t2, eps=1e-5):
    relative_diff = torch.norm(t2 - tref) / torch.norm(tref)
    assert relative_diff < eps


def angle(v1, v2):
    v1_flat = v1.get_flat_representation()
    v2_flat = v2.get_flat_representation()
    return torch.dot(v1_flat, v2_flat) / \
        (torch.norm(v1_flat) * torch.norm(v2_flat))
