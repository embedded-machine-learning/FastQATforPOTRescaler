# Generic Type imports
from asyncio.log import logger
from typing import Optional, Tuple, Union

# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_any_t, _size_2_t, Tensor, _size_any_opt_t


from .logger import logger_init,logger_forward



def FakeQuant(
    x: Tensor,
    delta_in: Tensor,
    delta_out: Tensor,
    training: bool,
    min_quant: Tensor,
    max_quant: Tensor,
    rounding_mode: str = "floor",
) -> Tensor:
    with torch.no_grad():
        if training:
            x.data.div_(delta_in, rounding_mode=rounding_mode).clamp_(min_quant, max_quant).mul_(delta_out)
        else:
            x = x.data.div(delta_in, rounding_mode=rounding_mode).clamp_(min_quant, max_quant)
    return x


def get_abs(self, x: torch.Tensor) -> torch.Tensor:
    if self.simple:
        abs = x.abs().max()
    else:
        xreorderd = x.permute(self.permutelist).contiguous()
        xreorderd = xreorderd.view((*xreorderd.shape[: self.numberofdims], -1))
        abs = xreorderd.abs().max(dim=(self.numberofdims), keepdim=True).values.view(self.size)
    return abs



class Quant(nn.Module):
    """
    Quant Quantization base module

    :param size: shape of the quantization delta
    :type size: tuple
    :param rounding_mode: how div should round, defaults to "floor"
    :type rounding_mode: str, optional
    """

    @logger_init
    def __init__(self, bits,size=(-1,), rounding_mode: str = "floor") -> None:
        super(Quant, self).__init__()
        self.simple = False
        self.bits = bits
        if size == (-1,):
            self.register_buffer("delta_in", torch.ones(1))
            self.register_buffer("delta_out", torch.ones(1))
            self.size = (1,)
            self.simple = True
        else:
            self.register_buffer("delta_in", torch.ones(size))
            self.register_buffer("delta_out", torch.ones(size))
            self.size = size

        self.permutelist = []
        self.reducelist = []
        self.numberofdims = 0
        for i in range(len(size)):
            # print(size[i])
            if size[i] != 1:
                self.permutelist.insert(0, i)
                self.numberofdims += 1
            else:
                self.permutelist.append(i)
                self.reducelist.append(i)

        self.permutelist = tuple(self.permutelist)
        self.rounding_mode = rounding_mode

        self.register_buffer("max", torch.ones(self.size)*(2 ** (self.bits - 1) - 1))
        self.register_buffer("min", torch.ones(self.size)*(-(2 ** (self.bits - 1))))
    
    @logger_forward
    def copy(self,other:'Quant'):
        self.delta_in = other.delta_in.clone().detach()
        self.delta_out = other.delta_out.clone().detach()
        self.min = other.min.clone().detach()
        self.max = other.max.clone().detach()
        self.rounding_mode = other.rounding_mode

    @logger_forward
    def forward(self, x:Tensor, fake:bool = False):
        if fake:
            return x
        return FakeQuant(
            x=x,
            delta_in=self.delta_in,
            delta_out=self.delta_out,
            training=self.training,
            min_quant=self.min,
            max_quant=self.max,
            rounding_mode=self.rounding_mode,
        )


class LinQuantExpScale(Quant):
    @logger_init
    def __init__(self, bits, size=(-1,), mom1=0.1, rounding_mode: str = "floor") -> None:
        super(LinQuantExpScale, self).__init__(bits,size, rounding_mode)
        if size == (-1,):
            self.register_buffer("abs", torch.ones(1))
        else:
            self.register_buffer("abs", torch.ones(size))
        self.take_new = True
        self.mom1 = mom1
        assert self.bits > 0
        self.register_buffer("delta_in_factor", torch.tensor(2.0 / (2.0**self.bits - 1)))
        self.register_buffer("delta_out_factor", torch.tensor(2.0 / (2.0**self.bits - 1)))

    @logger_forward
    def forward(self, x: torch.Tensor, fake:bool = False):
        if self.training:
            with torch.no_grad():
                abs = get_abs(self, x)
                # print(abs)
                self.abs = ((1 - self.mom1) * self.abs + self.mom1 * abs).detach()

                abs = self.abs.log2().ceil().exp2()
                self.delta_in = abs.mul(self.delta_in_factor).detach()  # .log2().ceil().exp2()
                self.delta_out = abs.mul(self.delta_out_factor).detach()  # .log2().ceil().exp2()

        return super(LinQuantExpScale, self).forward(x,fake)