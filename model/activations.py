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

    def forward(self, input: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        x, rexp = input
        LOG(__LOG_LEVEL_HIGH_DETAIL__, "LeakReLU.forward x", x)
        LOG(__LOG_LEVEL_HIGH_DETAIL__, "LeakReLU.forward rexp", rexp)
        if self.training:
            x = F.leaky_relu(x, negative_slope=self.negative_slope, inplace=self.inplace)
            LOG(__LOG_LEVEL_TO_MUCH__, "LeakReLU.forward x post relu", x)
            with torch.no_grad():
                x.data = x.data.div_(rexp.exp2(), rounding_mode="floor")
                LOG(__LOG_LEVEL_TO_MUCH__, "LeakReLU.forward x post scaling", x)
                x.data = x.data.mul_(rexp.exp2())
                LOG(__LOG_LEVEL_TO_MUCH__, "LeakReLU.forward x post scale-back", x)
        else:
            x = F.leaky_relu(x.type(torch.float32), negative_slope=self.negative_slope, inplace=self.inplace)
            LOG(__LOG_LEVEL_TO_MUCH__, "LeakReLU.forward x post relu", x)
            x = Floor.apply(x).type(torch.int32)
            LOG(__LOG_LEVEL_TO_MUCH__, "LeakReLU.forward x post floor", x)
        return x, rexp


class ReLU(nn.ReLU):
    """
    ReLU Encapsulation of nn.ReLU

    :param inplace:  if True inplace operations, defaults to False
    :type inplace: bool, optional
    """

    def __init__(self, inplace: bool = False):
        super().__init__(inplace)

    def forward(self, input: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[Tensor, Tensor]:
        x, rexp = input
        LOG(__LOG_LEVEL_HIGH_DETAIL__, "ReLU.forward x", x)
        LOG(__LOG_LEVEL_HIGH_DETAIL__, "ReLU.forward rexp", rexp)
        return super().forward(x), rexp


class PACT(nn.Module):
    """
    PACT The implementation of the PACT activation function

    This is the implementation of the PACT activation function from `https://openreview.net/forum?id=By5ugjyCb`

    :param size: The shape for alpha, defaults to (1,)
    :type size: tuple, optional
    """

    def __init__(self, size: tuple = (1,)) -> None:
        LOG(
            __LOG_LEVEL_DEBUG__,
            f"PACT arguments passed:\n\
            size:                           {size}\n\
            ",
        )
        super(PACT, self).__init__()
        self.size = size
        LOG(__LOG_LEVEL_HIGH_DETAIL__, "PACT.__init__: self.size", self.size)
        self.register_parameter("alpha", torch.nn.parameter(torch.ones(size)))
        LOG(__LOG_LEVEL_HIGH_DETAIL__, "PACT.__init__: parameter alpha", self.alpha)

    def forward(self, invals: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        val, rexp = invals
        if self.training:
            out = 0.5 * (val.abs() - (val - self.alpha).abs() + self.alpha)
            return out,rexp
        else:
            return val.clamp(min=0),rexp

class FusedActivation(nn.Module):
    """
    FusedActivation The base class for fused activations

    A fused activation will only be called in training, never in evaluation. It has do define a minimum and maximum.

    :param size: the shape of the minimum and maximum, defaults to (1,)
    :type size: tuple
    """
    def __init__(self,size = (1,)) -> None:
        super(FusedActivation,self).__init__()
        self.register_buffer("min",torch.zeros(size))
        self.register_buffer("max",torch.zeros(size))
    def forward(self,args):
        raise NotImplementedError()

class PACT_fused(FusedActivation):
    """
    PACT The implementation of the PACT activation function

    This is the implementation of the PACT activation function from `https://openreview.net/forum?id=By5ugjyCb`
    The fused part implicates, that is used inside the bn prior to quantizing, ans will never be called in the quantized domain

    :param size: The shape for alpha, defaults to (1,)
    :type size: tuple, optional
    """

    def __init__(self, size = (1,)) -> None:
        LOG(
            __LOG_LEVEL_DEBUG__,
            f"PACT arguments passed:\n\
            size:                           {size}\n\
            ",
        )
        super(PACT_fused, self).__init__(size)
        self.size = size
        LOG(__LOG_LEVEL_HIGH_DETAIL__, "PACT.__init__: self.size", self.size)
        self.register_parameter("alpha", torch.nn.Parameter(torch.ones(size)))
        LOG(__LOG_LEVEL_HIGH_DETAIL__, "PACT.__init__: parameter alpha", self.alpha)

        self.min = torch.zeros(size)

    def forward(self, val: Tensor) -> Tensor:
        assert self.training
        self.max = self.alpha.detach()
        out = 0.5 * (val.abs() - (val - self.alpha).abs() + self.alpha)
        return out
       

