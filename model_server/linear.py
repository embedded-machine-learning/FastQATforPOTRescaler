from numpy import empty
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Union

from model_old.quantizer import *
from model_old.utils import *


class LinQuant(nn.Linear):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None) -> None:
        super(LinQuant, self).__init__(in_features, out_features, False, device, dtype)

        self.quantw = LinQuantExpScale(8)
        self.register_buffer('used_weights', torch.zeros_like(self.weight))
        self.first_max = []
        self.take_new = True

    def forward(self, input: torch.Tensor, factor=1) -> torch.Tensor:
        # tmp = self.quantw(self.weight*factor)
        tmp = self.quantw(self.weight*factor)
        # print(torch.log2(self.quantw.desired_delta))
        # ~ 2**-9 is the scaling factor
        if not self.training:
            set_rexp(torch.round(torch.log2(self.quantw.delta))+get_rexp())
            tmp = tmp/self.quantw.delta
            self.used_weights = tmp
        if torch.any(torch.isnan(tmp)):
            print(torch.max(torch.abs(self.weight.data.view(-1))))
            print(factor)
        return F.linear(input, tmp, None)
