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

def tensortpr(tensor):
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

def _get_op_name(name, **kwargs):
    """
    Returns the operation's name that is performed in tpr format
    """
    return  'FP4_%s' % (name)
'''
def _gen_tpr_op(op, name, **kwargs):
    name = _get_op_name(name, **kwargs)
    tpr_args = unpack_bfp_args(kwargs)

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

            even,odd=tensortpr(grad_output, device)
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
def _get_tpr_op(op, name, tpr_args):
    """
    Create the tpr version of the operation op
    This function is called when a tpr layer is defined. See tprConv2d and tprLinear below
    """
    op_name = _get_op_name(name, **tpr_args)
    if op_name not in _tpr_ops:
        _tpr_ops[name] = _gen_tpr_op(op, name, tpr_args)

    return _tpr_ops[name]
'''

def unpack_tpr_args(kwargs):
    """
    Set up the tpr arguments
    """
    tpr_args = {}
    tpr_argn = [('num_format', 'fp32'),
                ('rounding_mode', 'stoc'),
                ('epsilon', 1e-8),
                ('init_grad_scale', 1.0)]

    for arg, default in tpr_argn:
        if arg in kwargs:
            tpr_args[arg] = kwargs[arg]
            del kwargs[arg]
        else:
            tpr_args[arg] = default
    return tpr_args

class _Scale_down(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, grad_scale):
        #print(f'_Scale_down forward input:{x}, {grad_scale}')
        ctx.grad_scale = grad_scale
        #print(f'_Scale_down forward output:{x / grad_scale}')
        return x / grad_scale

    @staticmethod
    def backward(ctx, grad):
        #print(f'_Scale_down backward input:{grad}')
        grad_scale = ctx.grad_scale
        #print(f'_Scale_down backward output:{grad / grad_scale}')
        return grad / grad_scale, None

class _Scale_up(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, grad_scale):
        #print(f'_Scale_up forward input:{x}, {grad_scale}')
        ctx.grad_scale = grad_scale
        #print(f'_Scale_up forward output:{x * grad_scale}')
        return x * grad_scale

    @staticmethod
    def backward(ctx, grad):
        #print(f'_Scale_up backward input:{grad}')
        grad_scale = ctx.grad_scale
        toret = grad * grad_scale

        g_scale = torch.tensor(0.0, requires_grad=False)
        if torch.max(toret)>64:
            g_scale = torch.tensor(-1.0, requires_grad=False)
        if torch.max(toret)<=32:
            g_scale = torch.tensor(1.0, requires_grad=False)

        #print(f'_Scale_up backward output:{grad * grad_scale} g_scale:{g_scale}')
        return toret, g_scale

class _TPR(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, bias=None, stride=1, padding=0, dilation=1, groups=1):
        #print(f'_TPR forward input:{x}, {w}')
        ctx.save_for_backward(x, w, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        out = F.conv2d(x, w, bias, stride, padding, dilation, groups)
        #print(f'_TPR forward output:{out}')
        return out


    @staticmethod
    def backward(ctx, grad_output):
        #print(f'_TPR backward input:{grad_output}')
        #pdb.set_trace()
        input, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        groups = ctx.groups
        grad_input = grad_weight = grad_bias = None

        even,odd=tensortpr(grad_output)
        grad_input = torch.nn.grad.conv2d_input(input.shape, weight, even, stride, padding, dilation, groups)
        grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, odd, stride, padding, dilation, groups)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = odd.sum((0,2,3)).squeeze(0)
        #print(f'_TPR backward output:{grad_input},{grad_weight},{grad_bias}')
        return grad_input, grad_weight, grad_bias, None, None, None, None
'''

class _TPR(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, bias=None):
        #print(f'_TPR forward input:{x}, {w}')
        ctx.save_for_backward(x, w, bias)
        out = x+w
        #print(f'_TPR forward output:{out}')
        return out


    @staticmethod
    def backward(ctx, grad_output):
        #print(f'_TPR backward input:{grad_output}')
        #pdb.set_trace()
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        grad_input = 1000+weight
        grad_weight = 2000+input
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = torch.tensor(66.5, requires_grad=True)
        #print(f'_TPR backward output:{grad_input},{grad_weight},{grad_bias}')
        return grad_input, grad_weight, grad_bias
'''

class TPRConv2d(torch.nn.Conv2d):
    """
    tpr convolutional layer
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, **kwargs)
        self.grad_scale = grad_scale

    def forward(self, input):
        #pdb.set_trace()
        #print(f'_TPR module forward input:{input} ')
        input = _Scale_down.apply(input, self.grad_scale)
        #print(f'_TPR module forward scaled down:{input} weight: {self.weight}')
        input = _TPR.apply(input, self.weight, self.bias)
        #print(f'_TPR module forward tpred:{input} weight: {self.weight}')
        input = _Scale_up.apply(input, self.grad_scale)
        #print(f'_TPR module forward scaled up:{input} weight: {self.weight}')

        return input

from torch.autograd.gradcheck import gradcheck
def test():
    dtype = torch.float
    device = torch.device("cuda:0")
    y_pred = TPRConv2d(4, 4, (3, 5), bias = None, stride=(2, 1), padding=(4, 2))
    optimizer = SGD(y_pred.parameters(), lr=0.1)

    y_pred.cuda()

    x = torch.randn(4, 4, 1, 2, device=device)
    y = torch.randn(4, 4, 28, 100, device=device)

    optimizer.zero_grad()

    pdb.set_trace()
    o = y_pred(x)
    loss = o.sum()
    loss.backward()
    moduleConv = TPRConv2d(3, 3)

    input = [torch.randn(20, 20, dtype=torch.double, requires_grad=True)]
    test = gradcheck(moduleConv, input, eps=1e-6, atol=1e-4)
    #print("Are the gradients correct: ", test)


def test_float_to_fp4():
    """
    Generate random fp32 tensors
    Convert them to tpr
    Check if the converted values are contained in the possible tpr numbers
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
    #print('first:')
    #print(end - start)

    #print(torch.tensor(finaleven).view(orig_shape))
    #print(torch.tensor(finalodd).view(orig_shape))
    t = torch.tensor(numbers).view(2,4,4)
    ###print(f't orig:{t}')
    orig_shape = t.size()
    t = t.view(-1)
    ###print(f't:{t}')
    quantized = []
    for i in t:
        i = tpr(i, epsilon, "even", device)
        quantized.append(i)
    ###print(f't:{t}')
    ###print(f'quantized:{quantized}')
    #print(final)
    #print(torch.tensor(quantized).view(orig_shape))
    '''

    for i in range(1000):
        x=1
    x_data = torch.tensor(numbers)
    start = time.time()
    for i in range(1000):
        e,o=tensortpr(x_data, epsilon, "even", device)
    end = time.time()
    ##print(f'first: {end - start}')
    ##print(f'even: {e}, odd: {o}')


import unittest
class TestAutograd(unittest.TestCase):
    def test_reentrant_priority(self):
        order = []

        class MyFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, x):
                order.append("MyFunction")
                return x

        class Reentrant(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                with torch.enable_grad():
                    ctx.x = torch.autograd.Variable(x.detach(), requires_grad=True)
                    ctx.x = ctx.x - 1
                return ctx.x.detach()

            @staticmethod
            def backward(ctx, x):
                order.append("Reentrant")
                if ctx.x < 0:
                    return x
                with torch.enable_grad():
                    Reentrant.apply(ctx.x).backward()
                return x

        a = MyFunction.apply(torch.tensor(6.0, requires_grad=True))
        b = Reentrant.apply(torch.tensor(9.0, requires_grad=True))
        v = a * b
        v.backward()
        # The tasks for the Reentrant and MyFunction backward() will be added
        # to the queue in the autograd engine at the same time. The backward
        # for Reentrant will be executed first, which will then add other
        # backward tasks to the queue. We want to ensure all the reentrant tasks
        # are prioritized over the MyFunction backward task regardless of their
        # sequence numbers
        #print(order)
        self.assertEqual(len(order), 11)
        self.assertEqual(order.count("Reentrant"), 10)
        self.assertEqual(order[-1], "MyFunction")

    def test_function_returns_input(self):
        class MyFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, grad):
                return grad * 2

        for shape in [(1,), ()]:
            v = torch.ones(shape, requires_grad=True)
            ##print(f'v: {v}')
            MyFunction.apply(v).backward()
            ##print(f'v: {v}')
            ##print(f'v.grad: {v.grad}')
            ##print(f'full: {torch.full(shape, 2.)}')
            self.assertEqual(v.grad, torch.full(shape, 2.))

            with torch.no_grad():
                v.grad.zero_()
            MyFunction.apply(v.clone()).backward()
            ##print(f'v: {v}')
            ##print(f'v.grad: {v.grad}')
            ##print(f'full: {torch.full(shape, 2.)}')
            self.assertEqual(v.grad, torch.full(shape, 2.))

def test_autograd():

    class down(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, grad_scale):
            #print(f'down forward input:{x}')
            ctx.grad_scale = grad_scale
            #print(f'down forward output:{x / grad_scale}')
            return x / grad_scale

        @staticmethod
        def backward(ctx, grad):
            #print(f'down backward input:{grad}')
            grad_scale = ctx.grad_scale
            #print(f'down backward output:{grad / grad_scale}')
            return grad / grad_scale, None

    class up(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, grad_scale):
            #print(f'up forward input:{x}')
            ctx.grad_scale = grad_scale
            #print(f'up forward output:{x * grad_scale}')
            return x * grad_scale

        @staticmethod
        def backward(ctx, grad):
            #print(f'up backward input:{grad}')
            grad_scale = ctx.grad_scale
            #print(f'up backward output:{grad * grad_scale}')
            return grad * grad_scale, None

    class tpr(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, w):
            ctx.save_for_backward(x, w)
            return x

        @staticmethod
        def backward(ctx, grad_output):
            #print(f'tpr backward input:{grad_output}')
            input, weight = ctx.saved_tensors
            grad_input = grad_weight = None

            grad_input = 100*weight
            grad_weight = 10*input
            #print(f'tpr backward output:{grad_input}, {grad_weight}')
            return grad_input, grad_weight


    class tpr2(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, w):
            ctx.save_for_backward(x, w)
            return x

        @staticmethod
        def backward(ctx, grad_output):
            #pdb.set_trace()
            #print(f'tpr2 backward input:{grad_output}')
            input, weight = ctx.saved_tensors
            grad_output.requires_grad = True
            with torch.enable_grad():
                '''
                grad_output = tpr.apply(grad_output, torch.tensor(1.0, requires_grad=False))
                ##print(f'grad_output:{grad_output}')
                grad_input , grad_weight = grad_output.backward()
                ##print(f'grad calc:{grad_input}, {grad_weight}')
                '''
                y = input * 5 + grad_output*62
                t = torch.autograd.grad(y, input)
                k = torch.autograd.grad(y, grad_output)
                ##print(f'grad calc:{y}, {input}, {grad_output}, {t}, {k}')

                grad_input = 100*grad_input
                grad_weight = 10*grad_weight
                #print(f'tpr2 backward output:{grad_input}, {grad_weight}')
                return grad_input, grad_weight

    input = torch.tensor(6.0, requires_grad=True)
    #print(f'main: {input}')
    input = down.apply(input, torch.tensor(3.0, requires_grad=True))
    #print(f'main down: {input}')
    input = tpr.apply(input, torch.tensor(5.0, requires_grad=True))
    #print(f'main tpr: {input}')
    input = up.apply(input, torch.tensor(3.0, requires_grad=True))
    #print(f'main up: {input}')
    input.backward()

def gradtest():
    class exampleFct(torch.autograd.Function):
        @staticmethod
        def forward(self, x):
            self.save_for_backward(x)
            return x ** 2

        @staticmethod
        def backward(self, dy):
            x, = self.saved_variables
            with torch.enable_grad():
                y = x ** 2
                return torch.autograd.grad(y, x, dy)


    x = torch.tensor(5.0, requires_grad=True)
    m = exampleFct.apply(x).sum().backward()
    #print(x.grad.data)


if __name__ == '__main__':
    #unittest.main(verbosity=2)
    #test_function_returns_input()
    #test_autograd()
    #gradtest()
    test_autograd()
