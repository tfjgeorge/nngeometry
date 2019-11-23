import torch


def check_ratio(vref, v2, eps=1e-3):
    ratio = v2 / vref
    assert ratio < 1 + eps and ratio > 1 - eps


def check_tensors(tref, t2, eps=1e-5):
    relative_diff = torch.norm(t2 - tref) / torch.norm(tref)
    assert relative_diff < eps
