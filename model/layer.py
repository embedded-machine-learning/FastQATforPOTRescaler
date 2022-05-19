import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch.nn.common_types import _size_any_t
from model.batchnorm import *

import numpy as np

from model.utils import *
from model.convolution import *


class Start(nn.Module):
    def __init__(self, running_exp_init) -> None:
        super(Start, self).__init__()
        self.register_buffer('run',torch.tensor([running_exp_init]))

    def forward(self, x):
        rexp=self.run
        x = x*(2**(-rexp[None,:,None,None]))
        x = Round.apply(x)
        if self.training:
            x = x/(2**(-rexp[None,:,None,None]))
        return (x, rexp)


class Stop(nn.Module):
    def __init__(self) -> None:
        super(Stop, self).__init__()
        self.size = []
        self.register_buffer('exp',None)
    def forward(self, invals: Tuple[torch.Tensor, torch.Tensor]):
        x , rexp = invals
        self.exp = rexp
        if not self.training:
            x = x/(2**-rexp[None,:,None,None])
        x = checkNan.apply(x)       # removes nan from backprop
        return x

class BlockQuant(nn.Module):
    def __init__(self, layers_in, layers_out, kernel_size, stride, groups=1) -> None:
        super(BlockQuant, self).__init__()

        self.conv = Conv2dLinChannelQuant(layers_in, layers_out, kernel_size, stride, padding=int(
            np.floor(kernel_size/2)), groups=groups)
        self.bn = BatchNorm2dQuant(layers_out)
        self.prelu = nn.LeakyReLU()

        self.first_old_exp = True
        self.old_exp = 0

    def forward(self, invals: Tuple[torch.Tensor, torch.Tensor]):
        
        fact = self.bn.get_weight_factor()

        # set sigma and old exp
        # if self.training:
        #    if self.first_old_exp:
        #        self.old_exp=get_rexp()
        #        self.first_old_exp=False
        #    self.bn.sig = self.bn.sig*(2**(2*(get_rexp()-self.old_exp)))
        #    self.old_exp=get_rexp()

        x = self.conv(invals, fact)
        x = self.bn(x, self.conv.quantw.delta)
       
        x , rexp = x
        x = self.prelu(x)

        if self.training:
            x = x*(2**(-rexp[None,:,None,None]))
            x = Round.apply(x)
            x = x/(2**(-rexp[None,:,None,None]))
        else:
            x = Round.apply(x)
        return x,rexp

class BlockQuantBiasChange(nn.Module):
    def __init__(self, layers_in, layers_out, kernel_size, stride, groups=1) -> None:
        super(BlockQuantBiasChange, self).__init__()

        self.conv = Conv2dLinChannelQuant(layers_in, layers_out, kernel_size, stride, padding=int(
            np.floor(kernel_size/2)), groups=groups)
        self.bn = BatchNorm2dQuantFixedBiasChange(layers_out)
        self.prelu = nn.LeakyReLU()

        self.first_old_exp = True
        self.old_exp = 0

    def forward(self, invals: Tuple[torch.Tensor, torch.Tensor]):
        
        fact = self.bn.get_weight_factor()

        # set sigma and old exp
        # if self.training:
        #    if self.first_old_exp:
        #        self.old_exp=get_rexp()
        #        self.first_old_exp=False
        #    self.bn.sig = self.bn.sig*(2**(2*(get_rexp()-self.old_exp)))
        #    self.old_exp=get_rexp()

        x = self.conv(invals, fact)
        x = self.bn(x, self.conv.quantw.delta)
       
        x , rexp = x
        x = self.prelu(x)

        if self.training:
            x = x*(2**(-rexp[None,:,None,None]))
            x = Round.apply(x)
            x = x/(2**(-rexp[None,:,None,None]))
        else:
            x = Round.apply(x)
        return x,rexp



class BlockQuant3(nn.Module):
    def __init__(self, layers_in, layers_out, kernel_size, stride, groups=1) -> None:
        super(BlockQuant3, self).__init__()

        self.conv = Conv2dLinChannelQuant(layers_in, layers_out, kernel_size, stride, padding=int(
            np.floor(kernel_size/2)), groups=groups)
        self.bn = BatchNorm2dQuantFixed(layers_out)
        self.prelu = nn.LeakyReLU()

        self.first_old_exp = True
        self.old_exp = 0

    def forward(self, invals: Tuple[torch.Tensor, torch.Tensor]):
        
        fact = self.bn.get_weight_factor()

        # set sigma and old exp
        # if self.training:
        #    if self.first_old_exp:
        #        self.old_exp=get_rexp()
        #        self.first_old_exp=False
        #    self.bn.sig = self.bn.sig*(2**(2*(get_rexp()-self.old_exp)))
        #    self.old_exp=get_rexp()

        x = self.conv(invals, fact)
        x = self.bn(x, self.conv.quantw.delta)
       
        x , rexp = x
        x = self.prelu(x)

        if self.training:
            x = x*(2**(-rexp[None,:,None,None]))
            x = Round.apply(x)
            x = x/(2**(-rexp[None,:,None,None]))
        else:
            x = Round.apply(x)
        return x,rexp


class BlockQuant4(nn.Module):
    def __init__(self, layers_in, layers_out, kernel_size, stride, groups=1) -> None:
        super(BlockQuant4, self).__init__()

        self.conv = Conv2dLinChannelQuant(layers_in, layers_out, kernel_size, stride, padding=int(
            np.floor(kernel_size/2)), groups=groups)
        self.bn = BatchNorm2dQuantFixedDynOut(layers_out)
        self.prelu = nn.LeakyReLU()

        self.first_old_exp = True
        self.old_exp = 0

    def forward(self, invals: Tuple[torch.Tensor, torch.Tensor]):
        
        fact = self.bn.get_weight_factor()

        # set sigma and old exp
        # if self.training:
        #    if self.first_old_exp:
        #        self.old_exp=get_rexp()
        #        self.first_old_exp=False
        #    self.bn.sig = self.bn.sig*(2**(2*(get_rexp()-self.old_exp)))
        #    self.old_exp=get_rexp()

        x = self.conv(invals, fact)
        x = self.bn(x, self.conv.quantw.delta)
       
        x , rexp = x
        x = self.prelu(x)

        if self.training:
            x = x*(2**(-rexp[None,:,None,None]))
            x = Round.apply(x)
            x = x/(2**(-rexp[None,:,None,None]))
        else:
            x = Round.apply(x)
        return x,rexp

class MaxPool(nn.MaxPool2d):
    def __init__(self, kernel_size: _size_any_t, stride: Optional[ _size_any_t] = None, padding:  _size_any_t = 0, dilation:  _size_any_t = 1, return_indices: bool = False, ceil_mode: bool = False) -> None:
        super(MaxPool,self).__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
    
    def forward(self, input: Tuple[torch.Tensor,torch.Tensor]):
        val,rexp = input
        return super().forward(val),rexp