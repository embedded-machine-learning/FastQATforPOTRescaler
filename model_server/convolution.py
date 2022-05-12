from numpy import empty
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Union

from model_server.quantizer import *
from model.utils import *


class Conv2dQuant(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1, padding: Union[str, _size_2_t] = 0, dilation: _size_2_t = 1, groups: int = 1, bias: bool = False, padding_mode: str = 'zeros', device=None, dtype=None) -> None:
        super(Conv2dQuant, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias, padding_mode, device, dtype)

        self.quantw = LinQuantExpScale(8,(out_channels,1,1,1))
        self.register_buffer('used_weights', torch.zeros_like(self.weight))
        self.first_max = []
        self.take_new = True

    def forward(self, input: torch.Tensor, factor=1) -> torch.Tensor:
        tmp = (self.weight) / \
            torch.sqrt(self.weight.var([1, 2, 3], unbiased=False)[:,None,None,None]+1e-5)
        # tmp = self.weight
        tmp = self.quantw(tmp*factor)
        # print(torch.log2(self.quantw.desired_delta))
        # ~ 2**-9 is the scaling factor
        if not self.training:
            set_rexp(torch.round(torch.log2(self.quantw.delta))+get_rexp())
            tmp = tmp/self.quantw.delta
            self.used_weights = tmp
        if torch.any(torch.isnan(tmp)):
            print(torch.max(torch.abs(self.weight.data.view(-1))))
            print(factor)
        return self._conv_forward(input, tmp, None)


class ChannelLinQuant(nn.Module):
    def __init__(self, bits, shape) -> None:
        super(ChannelLinQuant, self).__init__()
        self.bits = bits
        self.register_buffer('abs', torch.zeros(shape))
        self.take_new = True
        self.size = []
        self.register_buffer('delta', torch.ones(shape))

    def forward(self, x, fact=1):
        if len(self.size) == 0:
            self.size = list(x.shape)
            self.size[1] = -1
            for i in range(2, len(self.size)):
                self.size[i] = 1
        abs = torch.max(torch.abs(x.detach().view(self.size)),
                        dim=(1), keepdim=True).values

        if torch.any(abs < 1e-6):
            print("weights to small to quantize")

        abs = abs.masked_fill(abs < 1e-6, 1e-6)

        if self.take_new:
            self.abs = abs
            self.take_new = False
        elif self.training:
            self.abs = 0.9*self.abs + 0.1*abs

        self.delta = 2*(self.abs/(2.0**self.bits-1.0))
        return LinQuant_.apply(x*fact, self.abs, self.delta)


class Conv2dQuant2(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1, padding: Union[str, _size_2_t] = 0, dilation: _size_2_t = 1, groups: int = 1, bias: bool = False, padding_mode: str = 'zeros', device=None, dtype=None) -> None:
        super(Conv2dQuant2, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias, padding_mode, device, dtype)

        self.quantw = ChannelLinQuant(8,(out_channels,1,1,1))
        self.register_buffer('used_weights', torch.zeros_like(self.weight))
        self.first_max = []
        self.take_new = True

    def forward(self, input: torch.Tensor, factor=1) -> torch.Tensor:
        # tmp = (self.weight) / \
        #     torch.sqrt(self.weight.var([1, 2, 3], unbiased=False)[:,None,None,None]+1e-5)
        tmp = self.weight
        tmp = self.quantw(tmp,factor)
        # print(torch.log2(self.quantw.desired_delta))
        # ~ 2**-9 is the scaling factor
       
        if not self.training:
            tmp = tmp/self.quantw.delta
            self.used_weights = tmp
        # else:
        #     tmp=tmp/factor
        if torch.any(torch.isnan(tmp)):
            print(torch.max(torch.abs(self.weight.data.view(-1))))
            print(factor)
        return self._conv_forward(input, tmp, None)

