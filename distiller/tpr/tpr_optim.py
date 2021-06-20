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
from .tpr import unpack_tpr_args
import pdb

_tpr_optims = {}
def _gen_tpr_optim(optim, name):
    class TPROptim(optim):
        """
        Wrap the model's original optimizer in tpr

        Perform the original optimizer's  update function in fp32
        Convert the weights to two tpr formats: One with wide and another with narrow mantissas.
            Wide weights are used in future weight updates
            Narrow weights are used in forward and backward passes.
        """
        def __init__(self, *args, **kwargs):
            self.tpr_args = unpack_tpr_args(kwargs)
            super().__init__(*args, **kwargs)

        def step(self, *args, **kwargs):
            for group in self.param_groups:
                print(group)
                if(group['lr']==1):
                    print(f'grad group came: {group}')
            pdb.set_trace()
            # Apply step
            loss = super().step(*args, **kwargs)

            return loss

    TPROptim.__name__ = "TPR" + name
    return TPROptim


def get_tpr_optim(optim, name):
    if name not in _tpr_optims:
        _tpr_optims[name] = _gen_tpr_optim(optim, name)

    return _tpr_optims[name]
