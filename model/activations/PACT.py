import torch
import torch.nn as nn
from torch.nn.common_types import Tensor

from typing import Tuple

from ..logger import logger_init,logger_forward
from ..Quantizer import Quant


class PACT(Quant):
    """
    PACT The implementation of the PACT activation function

    This is the implementation of the PACT activation function from `https://openreview.net/forum?id=By5ugjyCb`
    
    :param size: The shape for alpha, defaults to (1,)
    :type size: tuple, optional
    """

    def __init__(self, bits, size=(-1,), rounding_mode: str = "floor") -> None:
        super(PACT, self).__init__(bits, size, rounding_mode)
        self.bits = bits
        assert self.bits > 0
        self.register_buffer("delta_in_factor", torch.tensor(1.0 / (2.0**self.bits - 1)))
        self.register_buffer("delta_out_factor", torch.tensor(1.0 / (2.0**self.bits - 1)))

        self.register_parameter("alpha", torch.nn.Parameter(6 * torch.ones(size)))
        self.register_buffer("alpha_used", torch.zeros_like(self.alpha))

        nn.init.constant_(self.min, 0)
        nn.init.constant_(self.max, 2**bits - 1)

    def forward(self, x: torch.Tensor, fake: bool = False):
        if self.training:
            with torch.no_grad():

                self.alpha_used = self.alpha.clone()  # block 2 small and negative alpha
                self.alpha_used = self.alpha_used.clamp(min=1e-3)  # block 2 small and negative alpha
                # abs = self.alpha_used.log2().ceil().exp2()
                self.delta_in = self.alpha_used.mul(self.delta_in_factor).detach()  # .log2().ceil().exp2()
                self.delta_out = self.alpha_used.mul(self.delta_out_factor).detach()  # .log2().ceil().exp2()

                self.max = self.alpha.div(self.delta_in, rounding_mode=self.rounding_mode).clamp(self.min)

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