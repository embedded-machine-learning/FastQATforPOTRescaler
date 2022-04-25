import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.activation import PReLUQuant

from model.convolution  import *
from model.batchnorm    import *
from model.utils        import *

class Start(nn.Module):
    def __init__(self,running_exp_init) -> None:
        super(Start,self).__init__()
        self.run=running_exp_init
    def forward(self,x):
        global running_exp
        running_exp = self.run
        return x
        
class Stop(nn.Module):
    def __init__(self) -> None:
        super(Stop,self).__init__()
    def forward(self,x):
        global running_exp
        if not self.train:
            return x*(2**running_exp)
        return x

class SplitConvBlockQuant(nn.Module):
    def __init__(self, layers_in, layers_out, kernel_size, stride) -> None:
        super(SplitConvBlockQuant, self).__init__()

        self.depthwise = BlockQuant(layers_in, layers_in, kernel_size, stride, layers_in)
        self.pointwise = BlockQuant(layers_in, layers_out, 1, 1, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class BlockQuant(nn.Module):
    def __init__(self, layers_in, layers_out, kernel_size, stride, groups=1) -> None:
        super(BlockQuant, self).__init__()

        self.conv = Conv2dQuant(layers_in, layers_out, kernel_size, stride, padding=int(np.floor(kernel_size/2)), groups=groups)
        self.bn = BatchNormQuant(layers_out)
        self.prelu = PReLUQuant(layers_out)

        self.first_old_exp = True
        self.old_exp=0

    def forward(self, x):
        global running_exp

        fact = self.bn.get_weight_factor()

        # set sigma and old exp
        if self.train:
            if self.first_old_exp:
                self.old_exp=running_exp
                self.first_old_exp=False
            self.bn.sig = self.bn.sig*(2**(2*(running_exp-self.old_exp)))
            self.old_exp=running_exp

        x = self.conv(x, fact)
        x = self.bn(x)

        x = self.prelu(x)
        if self.training:
            x = x*(2**(-running_exp))
            x = Round.apply(x)
            x = x/(2**(-running_exp))
        else:
            x = Round.apply(x)
        return x

class Block(nn.Module):
    def __init__(self, layers_in, layers_out, kernel_size, stride, groups=1) -> None:
        super(Block, self).__init__()

        self.conv = nn.Conv2d(layers_in, layers_out, kernel_size, stride, padding=int(np.floor(kernel_size/2)), groups=groups)
        self.bn = nn.BatchNorm2d(layers_out)
        self.prelu = nn.PReLU(layers_out)

    def forward(self, x):
        
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)

        return x
