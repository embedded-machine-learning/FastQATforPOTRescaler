from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import Tensor
from .utils import *


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


#################################################
#           FUNCTIONS                           #
#################################################


class expQuant(torch.autograd.Function):
    @staticmethod
    def forward(self, x):
        tmp = torch.exp2(torch.ceil(torch.log2(x)))
        # if torch.any(torch.isnan(tmp)):
        #     print("nan in exp scale")
        return tmp

    @staticmethod
    def backward(self, grad_output):
        # if torch.any(torch.isnan(grad_output)):
        #     print("nan in exp scale")
        return grad_output.detach()


class LinQuant_(torch.autograd.Function):
    @staticmethod
    def forward(self, x, abs, delta, bits):
        with torch.no_grad():
            # self.save_for_backward(x, abs)
            # x = x.clamp(-abs, abs)
            x = x.div_(delta, rounding_mode="floor").clamp_(-(2 ** (bits - 1)), 2 ** (bits - 1) - 1).mul_(delta)
            # if torch.any(torch.isnan(x)):
            #     print("nan in Linquant forward")
            return x

    @staticmethod
    def backward(self, grad_output: torch.Tensor):
        with torch.no_grad():
            # val = 2
            # x, abs = self.saved_tensors
            # grad_output = grad_output.masked_fill(torch.logical_and(
            #     torch.gt(x, val*abs), torch.gt(grad_output, 0)), 0)
            # grad_output = grad_output.masked_fill(torch.logical_and(
            #     torch.le(x, -val*abs), torch.le(grad_output, 0)), 0)
            # # if torch.any(torch.isnan(grad_output)):
            #     print("nan in Linquant back")
            return grad_output, None, None, None


class specialExpQuant(torch.autograd.Function):
    def forward(self, x: torch.Tensor):
        self.save_for_backward(x)
        x = x.clamp(0, 1)
        cond = torch.gt(x, 2**-6)
        n = torch.where(cond, torch.log2(expQuant.apply(x)), torch.zeros_like(x))
        x = torch.where(cond, expQuant.apply(x), torch.zeros_like(x))
        # if torch.any(torch.isnan(x)):
        #     print("nan in specialExpQuant forward")
        return x.detach(), n.detach(), cond.detach()

    def backward(self, grad_output: torch.Tensor, n_output: torch.Tensor, cond_output: torch.Tensor):
        (x,) = self.saved_tensors

        # only allow positive graients when greater 1 so that it can still get smaller
        grad_output = grad_output.masked_fill(torch.logical_and(torch.gt(x, 2.0), torch.gt(grad_output, 0)), 0)
        # only allow negative graients when smaller 2**-6 so that it can still get bigger
        grad_output = grad_output.masked_fill(
            torch.logical_and(torch.le(x, (2**-7) * 1.0), torch.le(grad_output, 0)), 0
        )
        if torch.any(torch.isnan(grad_output)):
            print("nan in specialExpQuant backward")
        return grad_output.detach(), None, None


def get_abs(self, x: torch.Tensor) -> torch.Tensor:
    if self.simple:
        abs = x.abs().max()
    else:
        xreorderd = x.permute(self.permutelist).contiguous()
        xreorderd = xreorderd.view((*xreorderd.shape[: self.numberofdims], -1))
        abs = xreorderd.abs().max(dim=(self.numberofdims), keepdim=True).values.view(self.size)
    return abs


def get_mean(self, x: torch.Tensor) -> torch.Tensor:
    if self.simple:
        abs = x.abs().mean()
    else:
        xreorderd = x.permute(self.permutelist).contiguous()
        xreorderd = xreorderd.view((*xreorderd.shape[: self.numberofdims], -1))
        abs = xreorderd.abs().mean(dim=(self.numberofdims), keepdim=True).view(self.size)
    return abs


#################################################
#           MODULES                             #
#################################################
class Filter_max(nn.Module):
    def __init__(self) -> None:
        super(Filter_max, self).__init__()
        self.past = []
        self.epoch_length = -1
        self.last_training = False
        self.i = 0
        self.last_dtype = None
        self.dev = None

    def forward(self, x: torch.Tensor):
        # return x
        mult = 1
        with torch.no_grad():
            if self.training:
                if self.last_dtype != x.dtype or x.device != self.dev:
                    print("Reseting Filter")
                    self.last_dtype = x.dtype
                    self.past = []
                    self.epoch_length = -1
                    self.i = 0
                    self.dev = x.device

                if not self.last_training:
                    self.last_training = True
                    self.i = 0

                if self.epoch_length == -1:
                    self.past.append(x.view(-1).detach().clone())
                else:
                    if self.i >= self.epoch_length:
                        print("incorrect epoch length")
                        self.past.append(x.view(-1).detach().clone())
                        self.epoch_length = -1
                    else:
                        self.past[self.i] = x.view(-1).detach().clone()
                        pass
                # if (self.epoch_length==-1 and self.i==0) or self.epoch_length==1:
                #     out = self.past[0]
                # else:
                out = torch.max(torch.stack(self.past, dim=1), dim=1, keepdim=True).values
                # print(out.shape)
                # print(x.shape)
                x.data = out.detach().clone().view(x.shape).to(x.device).type(x.dtype)
                self.i += 1
            else:
                if self.last_training:
                    self.last_training = False
                    if self.epoch_length == -1:
                        self.epoch_length = self.i
                        print("fixated length ")
                out = torch.max(torch.stack(self.past, dim=1), dim=1).values
                x.data = out.detach().clone().view(x.shape).to(x.device).type(x.dtype)
            return mult * x


class Filter_mean(nn.Module):
    def __init__(self) -> None:
        super(Filter_mean, self).__init__()
        self.past = []
        self.epoch_length = -1
        self.last_training = False
        self.i = 0
        self.last_dtype = None
        self.dev = None

    def forward(self, x: torch.Tensor):
        # return x
        mult = 1
        x = x.type(torch.float)
        with torch.no_grad():
            if self.training:
                if self.last_dtype != x.dtype or x.device != self.dev:
                    print("Reseting Filter")
                    self.last_dtype = x.dtype
                    self.past = []
                    self.epoch_length = -1
                    self.i = 0
                    self.dev = x.device

                if not self.last_training:
                    self.last_training = True
                    self.i = 0

                if self.epoch_length == -1:
                    self.past.append(x.view(-1).detach().clone())
                else:
                    if self.i >= self.epoch_length:
                        print("incorrect epoch length")
                        self.past.append(x.view(-1).detach().clone())
                        self.epoch_length = -1
                    else:
                        self.past[self.i] = x.view(-1).detach().clone()
                        pass
                # if (self.epoch_length==-1 and self.i==0) or self.epoch_length==1:
                #     out = self.past[0]
                # else:
                out = torch.mean(torch.stack(self.past, dim=1), dim=1, keepdim=True)
                # print(out.shape)
                # print(x.shape)
                x.data = out.detach().clone().view(x.shape).to(x.device).type(x.dtype)
                self.i += 1
            else:
                if self.last_training:
                    self.last_training = False
                    if self.epoch_length == -1:
                        self.epoch_length = self.i
                        print("fixated length ")
                out = torch.mean(torch.stack(self.past, dim=1), dim=1)
                x.data = out.detach().clone().view(x.shape).to(x.device).type(x.dtype)
            return mult * x


class Quant(nn.Module):
    """
    Quant Quantization base module

    :param size: shape of the quantization delta
    :type size: tuple
    :param rounding_mode: how div should round, defaults to "floor"
    :type rounding_mode: str, optional
    :param quant_int_dtype: The desired datatype, defaults to torch.int32
    :type quant_int_dtype: torch.dtype, optional
    """

    def __init__(self, size=Tuple, rounding_mode: str = "floor", quant_int_dtype=torch.int32) -> None:
        super(Quant, self).__init__()
        LOG(
            __LOG_LEVEL_DEBUG__,
            f"Quant passed arguments:\n\
            size:                           {size}\n\
            rounding_mode:                  {rounding_mode}\n\
            quant_int_dtype:                {quant_int_dtype}\n\
            ",
        )
        self.simple = False
        if size == (-1,):
            self.register_buffer("delta_in", torch.ones(1))
            self.register_buffer("delta_out", torch.ones(1))
            self.size = (1,)
            self.simple = True
        else:
            self.register_buffer("delta_in", torch.ones(size))
            self.register_buffer("delta_out", torch.ones(size))
            self.size = size

        LOG(__LOG_LEVEL_TO_MUCH__, "Quant.__init: buffer delta_in", self.delta_in)
        LOG(__LOG_LEVEL_TO_MUCH__, "Quant.__init: buffer delta_out", self.delta_out)
        LOG(__LOG_LEVEL_DEBUG__, "Quant.__init: simple", self.simple)

        self.permutelist = []
        self.numberofdims = 0
        for i in range(len(size)):
            # print(size[i])
            if size[i] != 1:
                self.permutelist.insert(0, i)
                self.numberofdims += 1
            else:
                self.permutelist.append(i)

        LOG(__LOG_LEVEL_DEBUG__, "Quant.__init: numberofdims", self.numberofdims)
        self.permutelist = tuple(self.permutelist)
        LOG(__LOG_LEVEL_DEBUG__, "Quant.__init: permutelist", self.permutelist)
        self.rounding_mode = rounding_mode
        LOG(__LOG_LEVEL_DEBUG__, "Quant.__init: rounding_mode", self.rounding_mode)
        self.quant_int_dtype = quant_int_dtype
        LOG(__LOG_LEVEL_DEBUG__, "Quant.__init: quant_int_dtype", self.quant_int_dtype)

    def forward(self, x):
        raise NotImplementedError()


class LinQuant(Quant):
    def __init__(self, bits, size=(-1,), mom1=0.1, rounding_mode: str = "floor", quant_int_dtype=torch.int32) -> None:
        super(LinQuant, self).__init__(size, rounding_mode, quant_int_dtype)
        self.bits = bits
        if size == (-1,):
            self.register_buffer("abs", torch.ones(1))
        else:
            self.register_buffer("abs", torch.ones(size))
        self.take_new = True
        self.mom1 = mom1
        assert self.bits > 0
        self.register_buffer("delta_in_factor", torch.tensor(2.0 / (2.0**self.bits - 1)))
        self.register_buffer("delta_out_factor", torch.tensor(2.0 / (2.0**self.bits - 1)))

        self.register_buffer("max", torch.tensor(2 ** (self.bits - 1) - 1))
        self.register_buffer("min", torch.tensor(-(2 ** (self.bits - 1))))

    def forward(self, x: torch.Tensor):
        if self.training:
            with torch.no_grad():
                abs = get_abs(self, x)
                # print(abs)
                self.abs = ((1 - self.mom1) * self.abs + self.mom1 * abs).detach()
                self.delta_in = self.abs.mul(self.delta_in_factor).detach()
                self.delta_out = self.abs.mul(self.delta_out_factor).detach()

        return FakeQuant(
            x=x,
            delta_in=self.delta_in,
            delta_out=self.delta_out,
            training=self.training,
            min_quant=self.min,
            max_quant=self.max,
            rounding_mode=self.rounding_mode,
            quant_int_dtype=self.quant_int_dtype,
        )


class LinQuantExpScale(Quant):
    def __init__(self, bits, size=(-1,), mom1=0.1, rounding_mode: str = "floor", quant_int_dtype=torch.int32) -> None:
        super(LinQuantExpScale, self).__init__(size, rounding_mode, quant_int_dtype)
        self.bits = bits
        if size == (-1,):
            self.register_buffer("abs", torch.ones(1))
        else:
            self.register_buffer("abs", torch.ones(size))
        self.take_new = True
        self.mom1 = mom1
        assert self.bits > 0
        self.register_buffer("delta_in_factor", torch.tensor(2.0 / (2.0**self.bits - 1)))
        self.register_buffer("delta_out_factor", torch.tensor(2.0 / (2.0**self.bits - 1)))

        self.register_buffer("max", torch.tensor(2 ** (self.bits - 1) - 1))
        self.register_buffer("min", torch.tensor(-(2 ** (self.bits - 1))))

    def forward(self, x: torch.Tensor, min=None, max=None):
        if self.training:
            with torch.no_grad():
                if min == None and max == None:
                    abs = get_abs(self, x)
                    # print(abs)
                    self.abs = ((1 - self.mom1) * self.abs + self.mom1 * abs).detach()

                    abs = self.abs.log2().ceil().exp2()
                    self.delta_in = abs.mul(self.delta_in_factor).detach()  # .log2().ceil().exp2()
                    self.delta_out = abs.mul(self.delta_out_factor).detach()  # .log2().ceil().exp2()
                else:
                    assert min != None
                    assert max != None
                    abs = torch.max(min.abs(),max.abs())

                    self.abs = ((1 - self.mom1) * self.abs + self.mom1 * abs).detach()
                    
                    abs = self.abs.log2().ceil().exp2()
                    self.delta_in = abs.mul(self.delta_in_factor).detach()  # .log2().ceil().exp2()
                    self.delta_out = abs.mul(self.delta_out_factor).detach()  # .log2().ceil().exp2()

                    self.min = min.div(self.delta_in,rounding_mode = self.rounding_mode)
                    self.max = max.div(self.delta_in,rounding_mode = self.rounding_mode)

        return FakeQuant(
            x=x,
            delta_in=self.delta_in,
            delta_out=self.delta_out,
            training=self.training,
            min_quant=self.min,
            max_quant=self.max,
            rounding_mode=self.rounding_mode,
            quant_int_dtype=self.quant_int_dtype,
        )


#########################################################################################################################
#                                    NEW STUFF                                                                          #
#########################################################################################################################

# class FakeQuant(torch.autograd.Function):
#     @staticmethod
#     def forward(self,x,factor):
#         with torch.no_grad():
#             return factor*torch.floor(x/factor)
#     def backward(self,grad):
#         return grad , None


def FakeQuant(
    x: Tensor,
    delta_in: Tensor,
    delta_out: Tensor,
    training: bool,
    min_quant: Tensor,
    max_quant: Tensor,
    rounding_mode: str = "floor",
    quant_int_dtype=torch.int32,
) -> Tensor:
    with torch.no_grad():
        if training:
            x.data.div_(delta_in, rounding_mode=rounding_mode).clamp_(min_quant, max_quant).mul_(delta_out)
        else:
            x = x.data.div(delta_in, rounding_mode=rounding_mode).clamp_(min_quant, max_quant).type(quant_int_dtype)
    return x


class LinQuant_new_(torch.autograd.Function):
    @staticmethod
    def forward(self, x, abs, delta, rexp_diff, fact):
        with torch.no_grad():
            x = x * (rexp_diff.exp2().view(1, -1, 1, 1)) * fact
            self.save_for_backward(x, abs)
            x = x.clamp(-abs, abs)
            x = x.div(delta, rounding_mode="floor").mul(delta)
            x = x / ((rexp_diff.exp2().view(1, -1, 1, 1)) * fact)
            if torch.any(torch.isnan(x)):
                print("nan in Linquant forward")
            return x

    @staticmethod
    def backward(self, grad_output: torch.Tensor):
        with torch.no_grad():
            val = 1
            x, abs = self.saved_tensors
            grad_output = grad_output.masked_fill(
                torch.logical_and(torch.gt(x, val * abs), torch.gt(grad_output, 0)), 0
            )
            grad_output = grad_output.masked_fill(
                torch.logical_and(torch.le(x, -val * abs), torch.le(grad_output, 0)), 0
            )
            if torch.any(torch.isnan(grad_output)):
                print("nan in Linquant back")
            return grad_output.detach(), None, None, None, None
