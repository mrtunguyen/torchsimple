from torch import nn
import torch
import numpy as np


def get_seq_lens(seq_len, conv):
    """
    Given a 1D Tensor or Variable containing integer sequence lengths, return a 1D tensor or variable containing 
    the size sequences that will be output by the network.
    :params seq_len: 1D tensor or int 
    :return: 1D Tensor scaled by model
    """
    def layer(x, first_dimension = False):
        if isinstance(x, tuple):
            if first_dimension:
                return x[0]
            else:
                return x[1]
        elif isinstance(x, int):
            return x

    if isinstance(seq_len, torch.Tensor):
        first_dimension = False
    else:
        first_dimension = True

    list_ = []
    for m in list(conv.children()):
        if type(m) != nn.Sequential:
            list_.append(m)
        else:
            for x in list(m.children()):
                for y in list(x.children()):
                    if type(y) != nn.Sequential:
                        list_.append(y)

    for m in list_:
        if type(m) == nn.modules.conv.Conv2d or type(m) == nn.modules.MaxPool2d:

            padding = layer(m.padding, first_dimension = first_dimension)
            dilation = layer(m.dilation, first_dimension = first_dimension)
            kernel_size = layer(m.kernel_size, first_dimension = first_dimension)
            stride = layer(m.stride, first_dimension = first_dimension)
            if first_dimension:
                seq_len = np.floor((seq_len + 2 * padding - dilation * (kernel_size - 1) -1) / stride + 1)
            else:
                seq_len = (np.floor((seq_len + 2 * padding - dilation * (kernel_size - 1) -1) / stride + 1)).int()

        elif type(m) == nn.modules.AvgPool2d:
            padding = layer(m.padding , first_dimension = first_dimension)
            kernel_size = layer(m.kernel_size, first_dimension = first_dimension)
            stride = layer(m.stride, first_dimension = first_dimension)
            if first_dimension:
                seq_len = np.floor((seq_len + 2 * padding - kernel_size)/stride + 1)
            else:
                seq_len = (np.floor((seq_len + 2 * padding - kernel_size)/stride + 1)).int()

    if isinstance(seq_len, torch.Tensor):
        seq_len[seq_len <= 0] = 1
        return seq_len.int()
    else:
        return int(seq_len)