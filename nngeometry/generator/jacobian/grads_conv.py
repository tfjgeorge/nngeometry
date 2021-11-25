# Author(s): Gaspar Rochette <gaspar.rochette@ens.fr>
# License: BSD 3 clause
# These functions are borrowed from https://github.com/owkin/grad-cnns

import numpy as np
import torch.nn as nn
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
    '''Computes per-example gradients for nn.Conv1d layers.

    This function is used in the internal behaviour of bnn.Linear.
    '''
    return conv_backward(*args, nd=1, **kwargs)


def conv2d_backward(mod, x, gy):
    '''Computes per-example gradients for nn.Conv2d layers.'''
    return conv_backward(x, gy, nd=2,
                         in_channels=mod.in_channels, 
                         out_channels=mod.out_channels,
                         kernel_size=mod.kernel_size,
                         stride=mod.stride,
                         dilation=mod.dilation,
                         padding=mod.padding,
                         groups=mod.groups)