# Generic Type imports
from typing import Tuple

# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import Tensor

# current module imports
from .utils import Floor
from .quantizer import FakeQuant, Quant, get_abs

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
            x = F.leaky_relu(x, negative_slope=self.negative_slope, inplace=self.inplace)
            LOG(__LOG_LEVEL_TO_MUCH__, "LeakReLU.forward x post relu", x)
            x = Floor.apply(x)
            LOG(__LOG_LEVEL_TO_MUCH__, "LeakReLU.forward x post floor", x)
        return x, rexp


class PACT_fused_F8NET_mod(Quant):
    """
    PACT The implementation of the PACT activation function forced to the quantization scheme of F8NET, it is *not* F8NETs implementation

    This is the implementation of the PACT activation function from `https://openreview.net/forum?id=By5ugjyCb`
    The fused part implicates, that is used inside other classes instead of the activation

    :param size: The shape for alpha, defaults to (1,)
    :type size: tuple, optional
    """

    def __init__(self, bits, size=(-1,), rounding_mode: str = "floor") -> None:
        super(PACT_fused_F8NET_mod, self).__init__(bits, size, rounding_mode)
        self.bits = bits
        assert self.bits > 0
        self.register_buffer("delta_in_factor", torch.tensor(1.0 / 70.0))
        self.register_buffer("delta_out_factor", torch.tensor(1.0 / 70.0))

        self.register_buffer("value_helper", torch.tensor(70.0 / (2**self.bits - 1)))
        self.register_buffer("max_helper", torch.tensor(2**self.bits - 1))

        self.register_parameter("alpha", torch.nn.Parameter(6 * torch.ones((1,1))))
        LOG(__LOG_LEVEL_HIGH_DETAIL__, "PACT.__init__: parameter alpha", self.alpha)
        self.register_buffer("alpha_used", torch.zeros_like(self.alpha))
        self.register_buffer("alpha_min", torch.Tensor([1e-3]))
        self.register_buffer("alpha_max", torch.Tensor([8]))


        nn.init.constant_(self.min, 0)
        nn.init.constant_(self.max, 2**bits - 1)

    def forward(self, x: torch.Tensor, fake: bool = False):
        if self.training:
            with torch.no_grad():
                # self.alpha_used = self.alpha.clone()  # block 2 small and negative alpha
                # self.alpha_used = self.alpha_used.clamp(min=1e-3)  # block 2 small and negative alpha

                # self.alpha_used= self.alpha.clamp(min=self.alpha_min,max = self.alpha_max)
                self.alpha_used= self.alpha.clamp(min=self.alpha_min)

                sigma = torch.var(x, self.reducelist, unbiased=False, keepdim=True).add(1e-5).sqrt()

                sigma = sigma.clamp(max=(self.alpha_used * self.value_helper))
                # sigma = self.alpha_used * self.value_helper

                self.delta_in = sigma.mul(self.delta_in_factor).log2().ceil().exp2().detach()       # floor became ceil bc. it is iverted pre log therefore times -1 therefore ceil
                self.delta_out = sigma.mul(self.delta_in_factor).log2().ceil().exp2().detach()

                self.max = self.alpha.div(self.delta_in, rounding_mode=self.rounding_mode).clamp(
                    min=self.min, max=self.max_helper
                )

            x = PACT_back_function.apply(x, self.alpha)
        return super().forward(x, fake)

