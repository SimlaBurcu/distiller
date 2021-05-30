# Copyright (c) 2021, Parallel Systems Architecture Laboratory (PARSA), EPFL &
# Machine Learning and Optimization Laboratory (MLO), EPFL. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the PARSA, EPFL & MLO, EPFL
#    nor the names of its contributors may be used to endorse or promote
#    products derived from this software without specific prior written
#    permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
import torch.nn.functional as F
from torch.optim import SGD
import numpy as np
import pdb
import itertools as it
import logging
import unittest
import math
import time

class rounding_modes:
    """
    When converting fp32 tensors to bfp, the rounding mode can be chosen.

    STOC: Stochastic rounding
    DETERM: Deterministic rounding
    """
    STOC, DETERM = 'stoc', 'determ'
    modes = [STOC, DETERM]

def round_tensor(t, mode, device):
    """
    Perform the rounding of the tensor t by using selected mode
    """
    if mode == rounding_modes.STOC:
        if device == "cpu":
            sampled = torch.FloatTensor(t.size(), device = device).uniform_(-0.5, 0.5)
        else:
            sampled = torch.cuda.FloatTensor(t.size()).uniform_(-0.5, 0.5)
        return sampled.add_(t).round()
    elif mode == rounding_modes.DETERM:
        return t.round()
    else:
        raise NotImplementedError("Rounding mode %s is not implemented", mode)

def get_exponent(t, epsilon):
    """
    Find the shared exponent of the tensor t.
    The exponent of the largest tensor value is selected as the shared exponent.
    """
    #Exponent is independent of the sign
    t = t.abs()
    #Find the maximum element of the tensor t
    max_v, _ = t.max(dim=1, keepdim=True)
    #Get the exponent of that element (We use ceil because in bfp format, we convert using 0.mantissa_bits instead of fp32's 1.mantissa_bits)
    return (max_v + epsilon).log2().ceil()

def _float_to_bfp(t, mant_bits, epsilon, rounding_mode, device, exp_given=None):
    """
    Convert float tensor t to bfp
    """
    exp = get_exponent(t, epsilon)

    #The interval between two consecutive numbers with that exponent value
    interval = torch.pow(2.0, exp-mant_bits)
    #The maximum representable value with exp
    max_v = torch.pow(2.0, exp) - interval

    # To ensure that we preserve the interval
    t = t/interval
    rounded = round_tensor(t, rounding_mode, device)
    rounded *=  interval

    #To ensure that there is no underflow or overflow
    return torch.min(torch.max(rounded, -max_v), max_v)

'''
def float_to_bfp_batched(t, mant_bits, epsilon, rounding_mode, device, bfp_tile_size=25,
                         num_format='', weight_mant_bits=''):
    """
    Convert a batch of fp32 tensor t to bfp
    """
    assert num_format == 'bfp'
    orig_shape = t.size()

    t = t.view(t.size()[0], -1)
    o = _float_to_bfp(t, mant_bits, epsilon, rounding_mode, device)
    return o.view(orig_shape)
'''
def float_to_bfp_batched(t, mant_bits, epsilon, rounding_mode, device, bfp_tile_size=25,
                         num_format='', weight_mant_bits='',backward=False):
    assert num_format == 'bfp'

    if backward:
        mant_bits = 7
    orig_shape = t.size()

    t = t.view(t.size()[0], -1)
    o = None
    o = _float_to_bfp(t, mant_bits, epsilon, rounding_mode, device)
    return o.view(orig_shape)

def tensor_to_tiled(t, orig_shape, bfp_tile_size):
    """
    Handle the tiling process.

    Output: the tiled tensor, the number of tiles in each dimension, the dimensions before and after the tiling to help 'untiling'
    """
    t = t.view(orig_shape[0], -1)
    matrix_h, matrix_w = t.size()

    numberOf_h_tiles = (matrix_h + bfp_tile_size - 1) // bfp_tile_size
    numberOf_w_tiles = (matrix_w + bfp_tile_size - 1) // bfp_tile_size

    matrix_h_pad = numberOf_h_tiles*bfp_tile_size
    matrix_w_pad = numberOf_w_tiles*bfp_tile_size

    h_pad = matrix_h_pad - matrix_h
    w_pad = matrix_w_pad - matrix_w

    t = F.pad(t, (0, w_pad, 0, h_pad),'constant')
    # t <-numberOf_h_tiles, tile_h, matrix_w
    t = t.view(numberOf_h_tiles, bfp_tile_size, matrix_w_pad)
    # t <- numberOf_h_tiles, matrix_w, tile_h,
    t.transpose_(1, 2)
    return (t.contiguous().view(numberOf_h_tiles*numberOf_w_tiles, -1),
            numberOf_h_tiles, numberOf_w_tiles,
            matrix_h, matrix_w,
            matrix_h_pad, matrix_w_pad)

def tiled_to_tensor(t, orig_shape, bfp_tile_size,
                    numberOf_h_tiles, numberOf_w_tiles,
                    matrix_h, matrix_w,
                    matrix_h_pad, matrix_w_pad):
    """
    Turn the tensor back to its shape before tiling
    """
    # t <- numberOf_h_tiles, numberOf_w_tiles, tile_w, tile_h
    t = t.view(numberOf_h_tiles, numberOf_w_tiles, bfp_tile_size, bfp_tile_size)
    # t <- numberOf_h_tiles, numberOf_w_tiles, tile_h, tile_w
    t.transpose_(2, 3)
    # t <- numberOf_h_tiles, tile_h, numberOf_w_tiles, tile_w
    t.transpose_(1, 2)
    t = t.contiguous().view(matrix_h_pad, matrix_w_pad)
    return t.narrow(0, 0, matrix_h).narrow(1, 0, matrix_w).view(orig_shape)


def float_to_bfp_tiled(t, mant_bits, epsilon, rounding_mode, device, bfp_tile_size=25,
                       num_format='', weight_mant_bits=0,
                       sgd_update=False, mant_bits_pow=None):
    """
    Convert fp32 tensor t to bfp with tiling.

    Used for weights (which are handled in the optimizer)
    """
    assert num_format == 'bfp'
    if sgd_update:
        mant_bits = weight_mant_bits

    orig_shape = t.size()
    if bfp_tile_size == 0:
        return _float_to_bfp(t.view(1, -1), mant_bits, epsilon, rounding_mode, device).view(orig_shape)

    (t, numberOf_h_tiles, numberOf_w_tiles, matrix_h, matrix_w,
        matrix_h_pad, matrix_w_pad) = tensor_to_tiled(t, orig_shape, bfp_tile_size)

    t = _float_to_bfp(t, mant_bits, epsilon, rounding_mode, device)

    return tiled_to_tensor(t, orig_shape, bfp_tile_size,
                           numberOf_h_tiles, numberOf_w_tiles,
                           matrix_h, matrix_w,
                           matrix_h_pad, matrix_w_pad)

def _get_op_name(name, epsilon, mant_bits, rounding_mode, **kwargs):
    """
    Returns the operation's name that is performed in BFP format
    """
    return  '%s_BFP_%s_%d' % (name, rounding_mode, mant_bits)

def _gen_bfp_op(op, name, bfp_args):
    """
    Do the 'sandwich'
    With an original op:

    out = op(x, y)
    grad_x, grad_y = op_grad(grad_out)

    To the following:
    x_, y_ = input_op(x, y)
    Where input_op(x, y) -> bfp(x), bfp(y)
    and input_op_grad(grad_x, grad_y) -> bfp(grad_x), bfp(grad_y)

    out_ = op(x_, y_)

    out = output_op(out)
    Where output_op(out) -> bfp(out)
    and output_op_grad(grad_out) -> bfp(grad_out)

    This way we garantee that everything in and out of the forward and backward operations is
    properly converted to bfp
    """

    name = _get_op_name(name, **bfp_args)

    class NewOpIn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, w):
            return (float_to_bfp_batched(x, **bfp_args, backward = False),
                    w)

        @staticmethod
        def backward(ctx, grad_x, grad_w):
            return (grad_x, grad_w)

    NewOpIn.__name__ = name + '_In'
    new_op_in = NewOpIn.apply

    class NewOpOut(torch.autograd.Function):
        @staticmethod
        def forward(ctx, op_out):
            return op_out

        @staticmethod
        def backward(ctx, op_out_grad):
            return float_to_bfp_batched(op_out_grad, **bfp_args, backward=True)

    NewOpOut.__name__ = name + '_Out'
    new_op_out = NewOpOut.apply

    def new_op(x, w, *args, **kwargs):
        x, w = new_op_in(x, w)
        out = op(x, w, *args, **kwargs)
        return new_op_out(out)

    return new_op

_bfp_ops = {}


def _get_bfp_op(op, name, bfp_args):
    """
    Create the bfp version of the operation op
    This function is called when a bfp layer is defined. See BFPConv2d and BFPLinear below
    """
    op_name = _get_op_name(name, **bfp_args)
    if op_name not in _bfp_ops:
        _bfp_ops[name] = _gen_bfp_op(op, name, bfp_args)

    return _bfp_ops[name]


def unpack_bfp_args(kwargs):
    """
    Set up the bfp arguments
    """
    bfp_args = {}
    bfp_argn = [('num_format', 'fp32'),
                ('rounding_mode', 'stoc'),
                ('epsilon', 1e-8),
                ('mant_bits', 0),
                ('bfp_tile_size', 0),
                ('weight_mant_bits', 0),
                ('device', 'cpu')]

    for arg, default in bfp_argn:
        if arg in kwargs:
            bfp_args[arg] = kwargs[arg]
            del kwargs[arg]
        else:
            bfp_args[arg] = default
    return bfp_args


def F_linear_bfp(**kwargs):
    """
    bfp linear function

    To be used in the model where F.linear is called
    """
    bfp_args = unpack_bfp_args(kwargs)
    if bfp_args['num_format'] == 'bfp':
        return _get_bfp_op(F.linear, 'linear', bfp_args)
    else:
        return F.linear

class BFPConv2d(torch.nn.Conv2d):
    """
    bfp convolutional layer
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, **kwargs):
        self.bfp_args = unpack_bfp_args(kwargs)

        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias)
        self.num_format = self.bfp_args['num_format']
        self.conv_op = _get_bfp_op(F.conv2d, 'Conv2d', self.bfp_args)

    def forward(self, input):
        if self.num_format == 'fp32':
            return F.conv2d(input, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

        elif self.num_format == 'bfp':
            conv = self.conv_op(input, self.weight, None, self.stride,
                                self.padding, self.dilation, self.groups)
            if self.bias is not None:
                return conv + self.bias
            else:
                return conv

        else:
            raise NotImplementedError('NumFormat not implemented')


class BFPLinear(torch.nn.Linear):
    """
    bfp linear layer
    """
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        self.bfp_args = unpack_bfp_args(kwargs)
        super().__init__(in_features, out_features, bias)
        self.num_format = self.bfp_args['num_format']
        self.linear_op = _get_bfp_op(F.linear, 'linear', self.bfp_args)

    def forward(self, input):
        if self.num_format == 'fp32':
            return F.linear(input, self.weight, self.bias)
        elif self.num_format == 'bfp':
            l = self.linear_op(input, self.weight, None)
            if self.bias is not None:
                return l + self.bias
            else:
                return l

        else:
            raise NotImplementedError('NumFormat not implemented')


def tensortpr2(tensor, epsilon, exp_given=None):
    """
    Convert float tensor t to fp4
    """
    zeros = torch.zeros_like(tensor)
    ones = torch.ones_like(tensor)
    sign = torch.where(tensor < 0, ones*-1, ones)
    t = torch.where(tensor == 0, zeros, tensor)
    t = t * 1.6
    log2t = torch.where(t == 0, zeros, t.abs().log2())
    ebit = log2t.floor()

    track = torch.zeros_like(tensor)
    ebit = (ebit / 2).floor()
    log2t = (log2t / 2)
    t = torch.where(ebit < -3, zeros, t)
    track = torch.where(ebit < -3, ones, track)
    t = torch.where(ebit >= 3, ones*64.0, t)
    track = torch.where(ebit >= 3, ones, track)
    ebit = ebit - torch.eq(ebit,log2t).int()
    t = torch.where(track == 0, torch.pow(4.0, ebit)*sign, t)
    even = torch.where(tensor == 0, zeros, t)


    t = torch.where(tensor == 0, zeros, tensor)
    t = t * 1.6
    log2t = torch.where(t == 0, zeros, t.abs().log2())
    ebit = log2t.floor()

    track = torch.zeros_like(tensor)
    t = torch.where(ebit < -7, zeros, t)
    track = torch.where(ebit < -7, ones, track)
    t = torch.where(ebit >= 5, ones*32.0, t)
    track = torch.where(ebit >= 5, ones, track)
    ebit = ebit - torch.eq(ebit%2,zeros).int()
    ebit = ebit - torch.eq(ebit,log2t).int()*2
    t = torch.where(track == 0, torch.pow(2.0, ebit)*sign, t)
    odd = torch.where(tensor == 0, zeros, t)
    return even,odd

def _gen_tpr_op(op, name):
    name = _get_op_name(name)

    class NewOpIn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, w):
            ctx.save_for_backward(input, weight)
            return (x, w)

        @staticmethod
        def backward(ctx, grad_x, grad_w):
            return (grad_x, grad_w)

    NewOpIn.__name__ = name + '_In'
    new_op_in = NewOpIn.apply

    class NewOpOut(torch.autograd.Function):
        @staticmethod
        def forward(ctx, op_out):
            return op_out

        @staticmethod
        def backward(ctx, op_out_grad):
            input, weight, bias = ctx.saved_tensors
            grad_input = grad_weight = grad_bias = None

            even,odd=tensortpr2(grad_output, epsilon, device)
            if ctx.needs_input_grad[0]:
                grad_input = even.mm(weight)
            if ctx.needs_input_grad[1]:
                grad_weight = odd.t().mm(input)
            if bias is not None and ctx.needs_input_grad[2]:
                grad_bias = odd.sum(0)
            return grad_input, grad_weight, grad_bias

    NewOpOut.__name__ = name + '_Out'
    new_op_out = NewOpOut.apply

    def new_op(x, w, *args, **kwargs):
        x, w = new_op_in(x, w)
        out = op(x, w, *args, **kwargs)
        return new_op_out(out)

    return new_op

_tpr_ops = {}
def _get_tpr_op(op, name):
    """
    Create the bfp version of the operation op
    This function is called when a bfp layer is defined. See BFPConv2d and BFPLinear below
    """
    op_name = _get_op_name(name)
    if op_name not in _tpr_ops:
        _tpr_ops[name] = _gen_tpr_op(op, name)

    return _tpr_ops[name]


class TPRConv2d(torch.nn.Conv2d):
    """
    tpr convolutional layer
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, **kwargs):

        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias)
        self.conv_op = _get_tpr_op(F.conv2d, 'Conv2d')

    def forward(self, input):
        conv = self.conv_op(input, self.weight, None, self.stride,
                                self.padding, self.dilation, self.groups)
        if self.bias is not None:
            return conv + self.bias
        else:
            return conv
        else:
            raise NotImplementedError('NumFormat not implemented')
def test():
    dtype = torch.float
    device = torch.device("cuda:0")
    y_pred = TPRConv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
    optimizer = SGD(y_pred.parameters(), lr=0.1)

    y_pred.cuda()

    for t in range(100):
        x = torch.randn(20, 16, 50, 100)
        y = torch.randn(20, 16, 50, 100)

        optimizer.zero_grad()

        o = y_pred(x)
        print(f'o: {o}')
        loss = (o - y).pow(2).sum()
        print(f'loss: {loss}')

        # Compute and print loss
        loss = loss.cuda()
        loss.backward()
        pdb.set_trace()
        optimizer.step()

        print(loss.item())

def test_float_to_fp4():
    """
    Generate random fp32 tensors
    Convert them to bfp
    Check if the converted values are contained in the possible bfp numbers
    """
    dtype = torch.float
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epsilon = 0
    rounding_mode = 'determ'

    numbers = [0.0,1.0,2.0,3.0,4.0,5.0,5.1,6.0,8.0,9.0,10.0,11.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0,21.0,22.0,23.0,24.0,25.0,26.0,27.0,28.0,29.0,30.0,31.0,32.0]
    numbers1 = [20.0,21.0,22.0,23.0,24.0,25.0,26.0,27.0,28.0,29.0,30.0,31.0,32.0,33.0,34.0,35.0,63.0,64.0]
    numbers2 = [0.0064, 0.00664, 0.01133, 0.5036, 0.3617, 0.43733, 0.09754, 0.1647]
    '''
    finaleven = []
    finalodd = []
    t = torch.tensor(numbers).view(2,4,4)
    start = time.time()
    orig_shape = t.size()
    t = t.view(-1)
    for n in t:
        c=tpr(n, epsilon, "even", device)
        d=tpr(n, epsilon, "odd", device)
        finaleven.append(c)
        finalodd.append(d)
    end = time.time()
    print('first:')
    print(end - start)

    print(torch.tensor(finaleven).view(orig_shape))
    print(torch.tensor(finalodd).view(orig_shape))
    t = torch.tensor(numbers).view(2,4,4)
    #print(f't orig:{t}')
    orig_shape = t.size()
    t = t.view(-1)
    #print(f't:{t}')
    quantized = []
    for i in t:
        i = tpr(i, epsilon, "even", device)
        quantized.append(i)
    #print(f't:{t}')
    #print(f'quantized:{quantized}')
    print(final)
    print(torch.tensor(quantized).view(orig_shape))
    '''

    for i in range(1000):
        x=1
    x_data = torch.tensor(numbers)
    start = time.time()
    for i in range(1000):
        e,o=tensortpr2(x_data, epsilon, "even", device)
    end = time.time()
    print(f'first: {end - start}')
    print(f'even: {e}, odd: {o}')


if __name__ == '__main__':
    #unittest.main(verbosity=2)
    test()
