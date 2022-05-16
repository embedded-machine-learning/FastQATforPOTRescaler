import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Union, Tuple

from model.quantizer import *
from model.utils import *

class Conv2dLayerLinQuant(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1, padding: Union[str, _size_2_t] = 0, dilation: _size_2_t = 1, groups: int = 1, bias: bool = False, padding_mode: str = 'zeros', device=None, dtype=None) -> None:
        super(Conv2dLayerLinQuant, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias, padding_mode, device, dtype)

        self.quantw = LinQuant(8)
        self.register_buffer('used_weights', torch.zeros_like(self.weight))

    def forward(self, invals: Tuple[torch.Tensor, torch.Tensor], factor=1) -> torch.Tensor:
        input, rexp = invals
        # tmp = (self.weight) / \
        #     torch.sqrt(self.weight.var([1, 2, 3], unbiased=False)[:,None,None,None]+1e-5)
        tmp = self.weight
        tmp = self.quantw(tmp,factor)
        rexp=torch.round(torch.log2(self.quantw.delta))+rexp
        if not self.training:
            tmp = tmp/self.quantw.delta
            self.used_weights = tmp

        if torch.any(torch.isnan(tmp)):
            print(torch.max(torch.abs(self.weight.data.view(-1))))
            print(factor)
        return self._conv_forward(input, tmp, None),rexp

class Conv2dLinChannelQuant(Conv2dLayerLinQuant):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1, padding: Union[str, _size_2_t] = 0, dilation: _size_2_t = 1, groups: int = 1, bias: bool = False, padding_mode: str = 'zeros', device=None, dtype=None) -> None:
        super(Conv2dLinChannelQuant, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias, padding_mode, device, dtype)

        self.quantw = LinQuant(8,(out_channels,1,1,1))

class Conv2dExpLayerQuant(Conv2dLayerLinQuant):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1, padding: Union[str, _size_2_t] = 0, dilation: _size_2_t = 1, groups: int = 1, bias: bool = False, padding_mode: str = 'zeros', device=None, dtype=None) -> None:
        super(Conv2dExpLayerQuant, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias, padding_mode, device, dtype)

        self.quantw = LinQuantExpScale(8)

class Conv2dExpChannelQuant(Conv2dLayerLinQuant):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1, padding: Union[str, _size_2_t] = 0, dilation: _size_2_t = 1, groups: int = 1, bias: bool = False, padding_mode: str = 'zeros', device=None, dtype=None) -> None:
        super(Conv2dExpLayerQuant, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias, padding_mode, device, dtype)

        self.quantw = LinQuantExpScale(8,(out_channels,1,1,1))