from pyexpat import model
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model_server.activation import PReLUQuant

from model_server.convolution import *
from model.utils import *
from model_server.linear import *
from model_server.batchnorm_fixed import *


class Start(nn.Module):
    def __init__(self, running_exp_init) -> None:
        super(Start, self).__init__()
        self.run = running_exp_init

    def forward(self, x):
        set_rexp(self.run)
        x = x*(2**(-get_rexp()))
        x = Round.apply(x)
        if self.training:
            x = x/(2**(-get_rexp()))
        return x


class Stop(nn.Module):
    def __init__(self, shape) -> None:
        super(Stop, self).__init__()
        self.size = []
        self.register_buffer('exponent', torch.ones(shape))

    def forward(self, x):
        if not self.training:
            if len(self.size)==0:
                self.size = list(x.shape)
                self.size[0]=1
                self.size[1]=-1
                for i in range(2,len(self.size)):
                    self.size[i]=1

            x = x/(2**-get_rexp().view(self.size))
            self.exponent = -get_rexp().view(self.size)
        return x


class Conv2dBN(nn.Module):
    def __init__(self, layers_in, layers_out, kernel_size, stride, groups=1) -> None:
        super(Conv2dBN, self).__init__()

        self.conv = Conv2dQuant(layers_in, layers_out, kernel_size, stride, padding=int(
            np.floor(kernel_size/2)), groups=groups)
        self.bn = BatchNorm2dQuant(layers_out)

    def forward(self, x):
        fact = self.bn.get_weight_factor()
        x = self.conv(x, fact)
        x = self.bn(x)
        return x


class LinearBN(nn.Module):
    def __init__(self, layers_in, layers_out) -> None:
        super(LinearBN, self).__init__()

        self.lin = LinQuant(layers_in, layers_out)
        self.bn = BatchNormQuant(layers_out)

    def forward(self, x):
        fact = self.bn.get_weight_factor()
        x = self.lin(x, fact)
        x = self.bn(x)
        return x


class SplitConvBlockQuant(nn.Module):
    def __init__(self, layers_in, layers_out, kernel_size, stride) -> None:
        super(SplitConvBlockQuant, self).__init__()

        self.depthwise = BlockQuant(
            layers_in, layers_in, kernel_size, stride, layers_in)
        self.pointwise = BlockQuant(layers_in, layers_out, 1, 1, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class SplitConvBlockQuant2(nn.Module):
    def __init__(self, layers_in, layers_out, kernel_size, stride) -> None:
        super(SplitConvBlockQuant2, self).__init__()

        self.depthwise = BlockQuant2(
            layers_in, layers_in, kernel_size, stride, layers_in)
        self.pointwise = BlockQuant2(layers_in, layers_out, 1, 1, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class BlockQuant(nn.Module):
    def __init__(self, layers_in, layers_out, kernel_size, stride, groups=1) -> None:
        super(BlockQuant, self).__init__()

        self.conv = Conv2dQuant(layers_in, layers_out, kernel_size, stride, padding=int(
            np.floor(kernel_size/2)), groups=groups)
        self.bn = BatchNorm2dQuant(layers_out)
        self.prelu = PReLUQuant(layers_out)

        self.first_old_exp = True
        self.old_exp = 0

    def forward(self, x):
        fact = self.bn.get_weight_factor()

        # set sigma and old exp
        # if self.training:
        #    if self.first_old_exp:
        #        self.old_exp=get_rexp()
        #        self.first_old_exp=False
        #    self.bn.sig = self.bn.sig*(2**(2*(get_rexp()-self.old_exp)))
        #    self.old_exp=get_rexp()

        x = self.conv(x, fact)
        x = self.bn(x)

        x = self.prelu(x)
        if self.training:
            x = x*(2**(-get_rexp()))
            x = Round.apply(x)
            x = x/(2**(-get_rexp()))
        else:
            x = Round.apply(x)
        return x

class BlockQuant2(nn.Module):
    def __init__(self, layers_in, layers_out, kernel_size, stride, groups=1) -> None:
        super(BlockQuant2, self).__init__()

        self.conv = Conv2dQuant2(layers_in, layers_out, kernel_size, stride, padding=int(
            np.floor(kernel_size/2)), groups=groups)
        self.bn = BatchNorm2dQuant2(layers_out)
        self.prelu = PReLUQuant(layers_out)

        self.first_old_exp = True
        self.old_exp = 0

    def forward(self, x):
        fact = self.bn.get_weight_factor(self.conv.quantw.delta)

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
            x = x*(2**(-get_rexp()))
            x = Round.apply(x)
            x = x/(2**(-get_rexp()))
        else:
            x = Round.apply(x)
        return x

class BlockQuant3(nn.Module):
    def __init__(self, layers_in, layers_out, kernel_size, stride, groups=1) -> None:
        super(BlockQuant3, self).__init__()

        self.conv = Conv2dQuant2(layers_in, layers_out, kernel_size, stride, padding=int(
            np.floor(kernel_size/2)), groups=groups)
        self.bn = BatchNorm2dQuantFixed(layers_out)
        self.prelu = nn.LeakyReLU()

        self.first_old_exp = True
        self.old_exp = 0

    def forward(self, x):
        fact = self.bn.get_weight_factor(self.conv.quantw.delta)

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
            x = x*(2**(-get_rexp()))
            x = Round.apply(x)
            x = x/(2**(-get_rexp()))
        else:
            x = Round.apply(x)
        return x

class Block(nn.Module):
    def __init__(self, layers_in, layers_out, kernel_size, stride, groups=1) -> None:
        super(Block, self).__init__()

        self.conv = nn.Conv2d(layers_in, layers_out, kernel_size, stride, padding=int(
            np.floor(kernel_size/2)), groups=groups)
        self.bn = nn.BatchNorm2d(layers_out)
        self.prelu = nn.ReLU()

    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)

        return x


class SplitBlock(nn.Module):
    def __init__(self, layers_in, layers_out, kernel_size, stride) -> None:
        super(SplitBlock, self).__init__()

        self.depthwise = Block(
            layers_in, layers_in, kernel_size, stride, layers_in)
        self.pointwise = Block(layers_in, layers_out, 1, 1, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
