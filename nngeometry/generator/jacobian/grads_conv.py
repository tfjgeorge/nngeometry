# Author(s): Gaspar Rochette <gaspar.rochette@ens.fr>
# License: BSD 3 clause
# These functions are borrowed from https://github.com/owkin/grad-cnns

import numpy as np
import torch
from torch._C import unify_type_list
import torch.nn.functional as F

def conv_backward(input, grad_output, in_channels, out_channels, kernel_size,
                  stride=1, dilation=1, padding=0, groups=1, nd=1):
    '''Computes per-example gradients for nn.Conv1d and nn.Conv2d layers.

    This function is used in the internal behaviour of bnn.Linear.
    '''

    # Change format of stride from int to tuple if necessary.
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * nd
    if isinstance(stride, int):
        stride = (stride,) * nd
    if isinstance(dilation, int):
        dilation = (dilation,) * nd
    if isinstance(padding, int):
        padding = (padding,) * nd

    # Get some useful sizes
    batch_size = input.size(0)
    input_shape = input.size()[-nd:]
    output_shape = grad_output.size()[-nd:]

    # Reshape to extract groups from the convolutional layer
    # Channels are seen as an extra spatial dimension with kernel size 1
    input_conv = input.view(1, batch_size * groups, in_channels // groups, *input_shape)

    # Compute convolution between input and output; the batchsize is seen
    # as channels, taking advantage of the `groups` argument
    grad_output_conv = grad_output.view(-1, 1, 1, *output_shape)

    stride = (1, *stride)
    dilation = (1, *dilation)
    padding = (0, *padding)

    if nd == 1:
        convnd = F.conv2d
        s_ = np.s_[..., :kernel_size[0]]
    elif nd == 2:
        convnd = F.conv3d
        s_ = np.s_[..., :kernel_size[0], :kernel_size[1]]
    elif nd == 3:
        raise NotImplementedError('3d convolution is not available with current per-example gradient computation')

    conv = convnd(
        input_conv, grad_output_conv,
        groups=batch_size * groups,
        stride=dilation,
        dilation=stride,
        padding=padding
    )

    # Because of rounding shapes when using non-default stride or dilation,
    # convolution result must be truncated to convolution kernel size
    conv = conv[s_]

    # Reshape weight gradient to correct shape
    new_shape = [batch_size, out_channels, in_channels // groups, *kernel_size]
    weight_bgrad = conv.view(*new_shape).contiguous()

    return weight_bgrad


def conv1d_backward(*args, **kwargs):
    '''Computes per-example gradients for nn.Conv1d layers.'''
    return conv_backward(*args, nd=1, **kwargs)


def conv2d_backward_using_conv(mod, x, gy):
    '''Computes per-example gradients for nn.Conv2d layers.'''
    return conv_backward(x, gy, nd=2,
                         in_channels=mod.in_channels, 
                         out_channels=mod.out_channels,
                         kernel_size=mod.kernel_size,
                         stride=mod.stride,
                         dilation=mod.dilation,
                         padding=mod.padding,
                         groups=mod.groups)


def conv2d_backward_using_unfold(mod, x, gy):
    '''Computes per-example gradients for nn.Conv2d layers.'''
    ks = (mod.weight.size(2), mod.weight.size(3))
    gy_s = gy.size()
    bs = gy_s[0]
    x_unfold = F.unfold(x, kernel_size=ks, stride=mod.stride,
                        padding=mod.padding, dilation=mod.dilation)
    x_unfold_s = x_unfold.size()
    return torch.bmm(gy.view(bs, gy_s[1], -1),
                     x_unfold.view(bs, x_unfold_s[1], -1).permute(0, 2, 1))


def conv2d_backward(*args, **kwargs):
    return _conv_grad_impl.get_impl()(*args, **kwargs)


class ConvGradImplManager:

    def __init__(self):
        self._use_unfold = True
    
    def use_unfold(self, choice=True):
        self._use_unfold = choice

    def get_impl(self):
        if self._use_unfold:
            return conv2d_backward_using_unfold
        else:
            return conv2d_backward_using_conv


_conv_grad_impl = ConvGradImplManager()

class use_unfold_impl_for_convs:

    def __enter__(self):
        self.prev = _conv_grad_impl._use_unfold
        _conv_grad_impl.use_unfold(True)

    def __exit__(self, exc_type, exc_value, traceback):
        _conv_grad_impl._use_unfold = self.prev

class use_conv_impl_for_convs:

    def __enter__(self):
        self.prev = _conv_grad_impl._use_unfold
        _conv_grad_impl.use_unfold(False)

    def __exit__(self, exc_type, exc_value, traceback):
        _conv_grad_impl._use_unfold = self.prev


def convtranspose2d_backward(mod, x, gy):
    '''Computes per-example gradients for nn.ConvTranspose2d layers.'''
    bs = gy.size(0)
    s_i, s_o, k_h, k_w = mod.weight.size()
    x_unfold = unfold_transpose_conv2d(mod, x)

    x_perm = x_unfold.view(bs, s_i*k_w*k_h, -1).permute(0, 2, 1)
    o = torch.bmm(gy.view(bs, s_o, -1), x_perm)
    o = o.view(bs, s_o, s_i, k_h, k_w).permute(0, 2, 1, 3, 4)
    o = o.contiguous()
    return o


def unfold_transpose_conv2d(mod, x):
    unfold_filter = _filter_bank.get(mod)
    return F.conv_transpose2d(x, unfold_filter, stride=mod.stride, padding=mod.padding,
                              output_padding=mod.output_padding, groups=mod.in_channels,
                              dilation=mod.dilation)

class TransposeConv_Unfold_Filter_Bank:

    def __init__(self):
        self.filters = dict()

    def get(self, mod):
        if mod not in self.filters:
            self.filters[mod] = self._create_unfold_filter(mod)
        return self.filters[mod]

    def _create_unfold_filter(self, mod):
        kw, kh = mod.kernel_size
        unfold_filter = mod.weight.data.new(mod.in_channels, kw * kh, kw, kh)
        unfold_filter.fill_(0)
        for i in range(mod.in_channels):
            for j in range(kw):
                for k in range(kh):
                    unfold_filter[i, k + kh*j, j, k] = 1
        return unfold_filter

_filter_bank = TransposeConv_Unfold_Filter_Bank()