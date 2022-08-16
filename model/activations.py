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


class ReLU_F8NET_fused(Quant):
    """
    ReLU_fused The implementation of the ReLU activation function fused into the quantization

    The fused part implicates, that is used instead of quantization

    :param size: The shape for alpha, defaults to (1,)
    :type size: tuple, optional
    """

    def __init__(self, bits, size=(-1,), mom1=0.1, rounding_mode: str = "floor", quant_int_dtype=torch.int32) -> None:
        super(ReLU_F8NET_fused, self).__init__(bits, size, rounding_mode, quant_int_dtype)
        self.bits = bits
        if size == (-1,):
            self.register_buffer("abs", torch.ones(1))
        else:
            self.register_buffer("abs", torch.ones(size))
        self.take_new = True
        self.mom1 = mom1
        assert self.bits > 0
        self.register_buffer("delta_in_factor", torch.tensor(1.0 / 70.0))
        self.register_buffer("delta_out_factor", torch.tensor(1.0 / 70.0))

        nn.init.constant_(self.min, 0)
        nn.init.constant_(self.max, 2**bits - 1)

    def forward(self, x: torch.Tensor, fake: bool = False):
        if self.training:
            with torch.no_grad():
                sigma = torch.var(x, [0, 2, 3], unbiased=False, keepdim=True).add(1e-5).sqrt()

                self.delta_in = sigma.mul(self.delta_in_factor).log2().floor().exp2().detach()
                self.delta_out = sigma.mul(self.delta_in_factor).log2().floor().exp2().detach()

            x = RELU_back_function.apply(x)
        return super().forward(x, fake)


class RELU_back_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, val: Tensor) -> Tensor:
        ctx.save_for_backward(val > 0)
        return val

    @staticmethod
    def backward(ctx, grad_outputs: Tensor) -> Tuple[Tensor, Tensor]:
        zero_cmp, = ctx.saved_tensors
        val_gard = grad_outputs * zero_cmp
        return val_gard


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
            return out, rexp
        else:
            return val.clamp(min=0), rexp


class FusedActivation(nn.Module):
    """
    FusedActivation The base class for fused activations

    A fused activation will only be called in training, never in evaluation. It has do define a minimum and maximum.

    :param size: the shape of the minimum and maximum, defaults to (1,)
    :type size: tuple
    """

    def __init__(self, size=(1,)) -> None:
        super(FusedActivation, self).__init__()
        self.register_buffer("min", torch.zeros(size))
        self.register_buffer("max", torch.zeros(size))

    def forward(self, args):
        raise NotImplementedError()


class PACT_fused(FusedActivation):
    """
    PACT The implementation of the PACT activation function

    This is the implementation of the PACT activation function from `https://openreview.net/forum?id=By5ugjyCb`
    The fused part implicates, that is used inside the bn prior to quantizing, ans will never be called in the quantized domain

    :param size: The shape for alpha, defaults to (1,)
    :type size: tuple, optional
    """

    def __init__(self, size=(1,)) -> None:
        LOG(
            __LOG_LEVEL_DEBUG__,
            f"PACT arguments passed:\n\
            size:                           {size}\n\
            ",
        )
        super(PACT_fused, self).__init__(size)
        self.size = size
        LOG(__LOG_LEVEL_HIGH_DETAIL__, "PACT.__init__: self.size", self.size)
        self.register_parameter("alpha", torch.nn.Parameter(6 * torch.ones(size)))
        LOG(__LOG_LEVEL_HIGH_DETAIL__, "PACT.__init__: parameter alpha", self.alpha)

        self.min = torch.zeros(size)

    def forward(self, val: Tensor) -> Tensor:
        assert self.training
        self.max = self.alpha.detach()
        out = 0.5 * (val.abs() - (val - self.alpha).abs() + self.alpha)
        return out


class PACT_fused_2(Quant):
    """
    PACT The implementation of the PACT activation function

    This is the implementation of the PACT activation function from `https://openreview.net/forum?id=By5ugjyCb`
    The fused part implicates, that is used inside the bn prior to quantizing, ans will never be called in the quantized domain

    :param size: The shape for alpha, defaults to (1,)
    :type size: tuple, optional
    """

    def __init__(self, bits, size=(-1,), rounding_mode: str = "floor", quant_int_dtype=torch.int32) -> None:
        super(PACT_fused_2, self).__init__(bits, size, rounding_mode, quant_int_dtype)
        self.bits = bits
        assert self.bits > 0
        self.register_buffer("delta_in_factor", torch.tensor(1.0 / (2.0**self.bits - 1)))
        self.register_buffer("delta_out_factor", torch.tensor(1.0 / (2.0**self.bits - 1)))

        self.register_parameter("alpha", torch.nn.Parameter(6 * torch.ones(size)))
        LOG(__LOG_LEVEL_HIGH_DETAIL__, "PACT.__init__: parameter alpha", self.alpha)
        self.register_buffer("alpha_used", torch.zeros_like(self.alpha))

        nn.init.constant_(self.min, 0)
        nn.init.constant_(self.max, 2**bits - 1)

    def forward(self, x: torch.Tensor, fake: bool = False):
        if self.training:
            with torch.no_grad():
                # abs = get_abs(self, x)
                # print(abs)
                # self.abs = ((1 - self.mom1) * self.abs + self.mom1 * abs).detach()
                self.alpha_used = self.alpha.clone()  # block 2 small and negative alpha
                self.alpha_used = self.alpha_used.clamp(min=1e-3)  # block 2 small and negative alpha
                abs = self.alpha_used.log2().ceil().exp2()
                self.delta_in = abs.mul(self.delta_in_factor).detach()  # .log2().ceil().exp2()
                self.delta_out = abs.mul(self.delta_out_factor).detach()  # .log2().ceil().exp2()

                self.max = self.alpha.div(self.delta_in, rounding_mode=self.rounding_mode).clamp(self.min)

            x = PACT_back_function.apply(x, self.alpha)
        return super().forward(x, fake)


class PACT_fused_F8NET_mod(Quant):
    """
    PACT The implementation of the PACT activation function forced to the quantization scheme of F8NET, it is *not* F8NETs implementation

    This is the implementation of the PACT activation function from `https://openreview.net/forum?id=By5ugjyCb`
    The fused part implicates, that is used inside other classes instead of the activation

    :param size: The shape for alpha, defaults to (1,)
    :type size: tuple, optional
    """

    def __init__(self, bits, size=(-1,), rounding_mode: str = "floor", quant_int_dtype=torch.int32) -> None:
        super(PACT_fused_F8NET_mod, self).__init__(bits, size, rounding_mode, quant_int_dtype)
        self.bits = bits
        assert self.bits > 0
        self.register_buffer("delta_in_factor", torch.tensor(1.0 / 70.0))
        self.register_buffer("delta_out_factor", torch.tensor(1.0 / 70.0))

        self.register_buffer("value_helper", torch.tensor(70.0 / (2**self.bits - 1)))
        self.register_buffer("max_helper", torch.tensor(2**self.bits - 1))

        self.register_parameter("alpha", torch.nn.Parameter(6 * torch.ones(size)))
        LOG(__LOG_LEVEL_HIGH_DETAIL__, "PACT.__init__: parameter alpha", self.alpha)
        self.register_buffer("alpha_used", torch.zeros_like(self.alpha))
        self.register_buffer("clamp_min", torch.Tensor([1e-3]))


        nn.init.constant_(self.min, 0)
        nn.init.constant_(self.max, 2**bits - 1)

    def forward(self, x: torch.Tensor, fake: bool = False):
        if self.training:
            with torch.no_grad():
                # self.alpha_used = self.alpha.clone()  # block 2 small and negative alpha
                # self.alpha_used = self.alpha_used.clamp(min=1e-3)  # block 2 small and negative alpha

                self.alpha_used= self.alpha.clamp(min=self.clamp_min)

                sigma = torch.var(x, [0, 2, 3], unbiased=False, keepdim=True).sqrt()

                sigma = sigma.clamp(max=(self.alpha_used * self.value_helper))

                self.delta_in = sigma.mul(self.delta_in_factor).log2().ceil().exp2().detach()       # floor became ceil bc. it is iverted pre log therefore times -1 therefore ceil
                self.delta_out = sigma.mul(self.delta_in_factor).log2().ceil().exp2().detach()

                self.max = self.alpha.div(self.delta_in, rounding_mode=self.rounding_mode).clamp(
                    min=self.min, max=self.max_helper
                )

            x = PACT_back_function.apply(x, self.alpha)
        return super().forward(x, fake)

class PACT_back_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, val: Tensor, alpha: Tensor) -> Tensor:
        ctx.save_for_backward(val >= alpha, val > 0)
        return val

    @staticmethod
    def backward(ctx, grad_outputs: Tensor) -> Tuple[Tensor, Tensor]:
        alpha_cmp, zero_cmp = ctx.saved_tensors
        val_gard = grad_outputs * zero_cmp * (~alpha_cmp)
        alpha_grad = grad_outputs * alpha_cmp * zero_cmp
        return val_gard, alpha_grad
