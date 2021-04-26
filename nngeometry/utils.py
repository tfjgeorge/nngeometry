import torch
import torch.nn.functional as F
from nngeometry.object.vector import PVector


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


def grad(output, vec, *args, **kwargs):
    """
    Computes the gradient of `output` with respect to the `PVector` `vec`

    ..warning This function only works when internally your `vec` has been
        created from leaf nodes in the graph (e.g. model parameters)
    
    :param output: The scalar quantity to be differentiated
    :param vec: a `PVector`
    :return: a `PVector` of gradients of `output` w.r.t `vec`
    """
    if vec.dict_repr is not None:
        # map all parameters to a list
        params = []
        pos = []
        lenghts = []
        current_pos = 0
        for k in vec.dict_repr.keys():
            p = vec.dict_repr[k]
            params += list(p)
            pos.append(current_pos)
            lenghts.append(len(p))
            current_pos = current_pos + len(p)

        grad_list = torch.autograd.grad(output, params, *args, **kwargs)
        dict_repr_grad = dict()

        for k, p, l in zip(vec.dict_repr.keys(), pos, lenghts):
            if l == 1:
                dict_repr_grad[k] = (grad_list[p],)
            elif l == 2:
                dict_repr_grad[k] = (grad_list[p], grad_list[p+1])

        return PVector(vec.layer_collection,
                       dict_repr=dict_repr_grad)
    else:
        raise RuntimeError('grad only works with the vector is created ' +
                           'from leaf nodes in the computation graph')
