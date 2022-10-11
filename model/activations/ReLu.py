from typing import Tuple

import torch
import torch.nn as nn
from torch.nn.common_types import Tensor

from ..Quantizer import Quant

class ReLU(Quant):
    """
    ReLU The implementation of the ReLU activation function fused into the quantization

    :param size: The shape for alpha, defaults to (1,)
    :type size: tuple, optional
    """

    def __init__(self, bits, size=(-1,), mom1=0.1, rounding_mode: str = "floor") -> None:
        super(ReLU, self).__init__(bits, size, rounding_mode)
        self.bits = bits
        if size == (-1,):
            self.register_buffer("sigma", torch.ones(1))
        else:
            self.register_buffer("sigma", torch.ones(size))
        self.take_new = True
        self.mom1 = mom1
        assert self.bits > 0
        # as defined by f8net
        self.register_buffer("delta_in_factor", torch.tensor(1.0 / 70.0))
        self.register_buffer("delta_out_factor", torch.tensor(1.0 / 70.0))

        nn.init.constant_(self.min, 0)
        nn.init.constant_(self.max, 2**bits - 1)

    def forward(self, x: torch.Tensor, fake: bool = False):
        if self.training:
            with torch.no_grad():
                if len(self.size) != len(x.shape):
                    self.size = self.size + [ 1 for x in range(len(x.shape)-len(self.size)) ]
                    super(ReLU,self).__init__(self.bits,self.size,self.rounding_mode)
                    print("mismatch in input and definition found, louding wont be possiblöe until fixed")

                sigma = torch.var(x, self.reducelist, unbiased=False, keepdim=True).add(1e-5).sqrt()
                if self.take_new:
                    self.take_new = False
                    self.sigma = sigma
                else:
                    self.sigma = (1-self.mom1) * self.sigma + self.mom1 * sigma

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
