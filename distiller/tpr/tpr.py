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

def _get_op_name(name):
    """
    Returns the operation's name that is performed in BFP format
    """
    return  'FP4_%s' % (name)

def _gen_tpr_op(op, name):
    name = _get_op_name(name)

    class NewOpIn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, w):
            ctx.save_for_backward(x, w)
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
            input, weight = ctx.saved_tensors
            grad_input = grad_weight = grad_bias = None

            even,odd=tensortpr2(grad_output, epsilon, device)
            if ctx.needs_input_grad[0]:
                grad_input = even.mm(weight)
            if ctx.needs_input_grad[1]:
                grad_weight = odd.t().mm(input)
            return grad_input, grad_weight

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
def test():
    dtype = torch.float
    device = torch.device("cuda:0")
    y_pred = TPRConv2d(16, 33, (3, 5), bias = None, stride=(2, 1), padding=(4, 2))
    optimizer = SGD(y_pred.parameters(), lr=0.1)

    y_pred.cuda()

    for t in range(100):
        x = torch.randn(20, 16, 50, 100, device=device)
        y = torch.randn(20, 33, 28, 100, device=device)

        optimizer.zero_grad()

        pdb.set_trace()
        o = y_pred(x)
        print(f'o: {o}')
        loss = (o - y).pow(2).sum()
        print(f'loss: {loss}')

        # Compute and print loss
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
