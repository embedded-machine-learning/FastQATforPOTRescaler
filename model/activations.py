# Generic Type imports
from typing import Tuple

# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import Tensor

# current module imports
from .utils import Floor

# Global information imports
from . import (
    __DEBUG__,
    LOG,
    __LOG_LEVEL_IMPORTANT__,
    __LOG_LEVEL_NORMAL__,
    __LOG_LEVEL_DEBUG__,
    __LOG_LEVEL_HIGH_DETAIL__,
    __LOG_LEVEL_TO_MUCH__,
)

class LeakReLU(torch.nn.LeakyReLU):
    """
    LeakReLU Encapsulated nn.LeakReLU


    :param negative_slope: the negative slope (use pow of 2 ), defaults to 2**-6
    :type negative_slope: float, optional
    :param inplace: if True inplace operations, defaults to False
    :type inplace: bool, optional
    """

    def __init__(self, negative_slope: float = 2**-6, inplace: bool = False) -> None:
        LOG(
            __LOG_LEVEL_DEBUG__,
            f"LeakReLU passed arguments:\n\
            negative_slope:             {negative_slope}\n\
            inplace:                    {inplace}",
        )
        super().__init__(negative_slope, inplace)

    def forward(self, input: Tuple[Tensor, Tensor]):
        x, rexp = input
        LOG(__LOG_LEVEL_DEBUG__,"LeakReLU.forward x",x)
        LOG(__LOG_LEVEL_DEBUG__,"LeakReLU.forward rexp",rexp)
        if self.training:
            x = F.leaky_relu(x, negative_slope=self.negative_slope, inplace=self.inplace)
            LOG(__LOG_LEVEL_TO_MUCH__,"LeakReLU.forward x post relu",x)
            x = x * (2 ** (-rexp.view(-1)[None, :, None, None]))
            LOG(__LOG_LEVEL_TO_MUCH__,"LeakReLU.forward x post scaling",x)
            x = Floor.apply(x)
            LOG(__LOG_LEVEL_TO_MUCH__,"LeakReLU.forward x post floor",x)
            x = x / (2 ** (-rexp.view(-1)[None, :, None, None]))
            LOG(__LOG_LEVEL_TO_MUCH__,"LeakReLU.forward x post scale-back",x)
        else:
            x = F.leaky_relu(x.type(torch.float32), negative_slope=self.negative_slope, inplace=self.inplace)
            LOG(__LOG_LEVEL_TO_MUCH__,"LeakReLU.forward x post relu",x)
            x = Floor.apply(x).type(torch.int32)
            LOG(__LOG_LEVEL_TO_MUCH__,"LeakReLU.forward x post floor",x)
        return x, rexp


class ReLU(torch.nn.ReLU):
    """
    ReLU Encapsulation of nn.ReLU

    :param inplace:  if True inplace operations, defaults to False
    :type inplace: bool, optional
    """
    def __init__(self, inplace: bool = False):
        super().__init__(inplace)

    def forward(self, input: Tuple[torch.Tensor, torch.Tensor]):
        x, rexp = input
        LOG(__LOG_LEVEL_DEBUG__,"ReLU.forward x",x)
        LOG(__LOG_LEVEL_DEBUG__,"ReLU.forward rexp",rexp)
        return super().forward(x), rexp
