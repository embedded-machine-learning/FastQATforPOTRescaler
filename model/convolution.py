import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Union

from model.quantizer import *
from model.utils import *


class Conv2dQuant(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1, padding: Union[str, _size_2_t] = 0, dilation: _size_2_t = 1, groups: int = 1, bias: bool = False, padding_mode: str = 'zeros', device=None, dtype=None) -> None:
        super(Conv2dQuant, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias, padding_mode, device, dtype)

        self.quantw = LinQuantExpScale(8)
        self.register_buffer('used_weights', torch.zeros_like(self.weight))
        self.first_max = []
        self.take_new = True

    def forward(self, input: torch.Tensor, factor=1) -> torch.Tensor:
        # tmp = (self.weight) / \
        #     torch.sqrt(self.weight.var([1, 2, 3], unbiased=False)[:,None,None,None]+1e-5)
        tmp = self.weight
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
