import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from model.batchnorm import BatchNorm2dQuantFixed

import numpy as np

from model.utils import *
from model.convolution import *


class Start(nn.Module):
    def __init__(self, running_exp_init) -> None:
        super(Start, self).__init__()
        self.register_buffer('run',torch.tensor(running_exp_init))

    def forward(self, x):
        rexp=self.run
        x = x*(2**(-rexp))
        x = Round.apply(x)
        if self.training:
            x = x/(2**(-rexp))
        return (x, rexp)


class Stop(nn.Module):
    def __init__(self) -> None:
        super(Stop, self).__init__()
        self.size = []
    def forward(self, invals: Tuple[torch.Tensor, torch.Tensor]):
        x , rexp = invals
        if not self.training:
            if len(self.size)==0:
                self.size = list(x.shape)
                self.size[0]=1
                self.size[1]=-1
                for i in range(2,len(self.size)):
                    self.size[i]=1

            x = x/(2**-rexp.view(self.size))
        return x



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
        x , rexp = invals
        fact = self.bn.get_weight_factor()

        # set sigma and old exp
        # if self.training:
        #    if self.first_old_exp:
        #        self.old_exp=get_rexp()
        #        self.first_old_exp=False
        #    self.bn.sig = self.bn.sig*(2**(2*(get_rexp()-self.old_exp)))
        #    self.old_exp=get_rexp()

        x = self.conv(x, fact)
        x = self.bn(x, self.conv.quantw.delta)

        x = self.prelu(x)
        if self.training:
            x = x*(2**(-rexp()))
            x = Round.apply(x)
            x = x/(2**(-rexp()))
        else:
            x = Round.apply(x)
        return x,rexp