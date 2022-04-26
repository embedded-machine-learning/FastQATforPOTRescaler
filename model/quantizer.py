from asyncio import trsock
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F


#################################################
#           FUNCTIONS                           #
#################################################


class expQuant(torch.autograd.Function):
    @staticmethod
    def forward(self, x):
        return torch.exp2(torch.round(torch.log2(x)))

    @staticmethod
    def backward(self, grad_output):
        return grad_output.detach()


class LinQuant_(torch.autograd.Function):
    @staticmethod
    def forward(self, x, abs, delta):
        self.save_for_backward(x, abs)
        x = torch.clamp(x, -abs, abs)
        x = torch.round((x)/delta)*delta

        return x

    @staticmethod
    def backward(self, grad_output: torch.Tensor):
        x, abs = self.saved_tensors
        grad_output = grad_output.masked_fill(torch.logical_and(
            torch.gt(x, 2*abs), torch.gt(grad_output, 0)), 0)
        grad_output = grad_output.masked_fill(torch.logical_and(
            torch.le(x, -2*abs), torch.le(grad_output, 0)), 0)
        return grad_output.detach(), None, None


class specialExpQuant(torch.autograd.Function):
    def forward(self, x: torch.Tensor):
        self.save_for_backward(x)
        x = x.clamp(0, 1)
        cond = torch.gt(x, 2**-6)
        n = torch.where(cond, torch.log2(
            expQuant.apply(x)), torch.zeros_like(x))
        x = torch.where(cond, expQuant.apply(x), torch.zeros_like(x))
        return x.detach(), n.detach(), cond.detach()

    def backward(self, grad_output: torch.Tensor, n_output: torch.Tensor, cond_output: torch.Tensor):
        x, = self.saved_tensors

        # only allow positive graients when greater 1 so that it can still get smaller
        grad_output = grad_output.masked_fill(torch.logical_and(
            torch.gt(x, 2.0), torch.gt(grad_output, 0)), 0)
        # only allow negative graients when smaller 2**-6 so that it can still get bigger
        grad_output = grad_output.masked_fill(torch.logical_and(
            torch.le(x, (2**-7)*1.0), torch.le(grad_output, 0)), 0)
        return grad_output.detach(), None, None



#################################################
#           MODULES                             #
#################################################
class Quant(nn.Module):
    def __init__(self) -> None:
        super(Quant, self).__init__()
        self.register_buffer('delta', torch.zeros(1))

    def forward(self, x):
        raise NotImplementedError()


class LinQuant(Quant):
    def __init__(self, bits) -> None:
        super(LinQuant, self).__init__()
        self.bits = bits
        self.register_buffer('abs', torch.zeros(1))
        self.take_new = True

    def forward(self, x):
        abs = torch.max(torch.abs(x.detach().view(-1)))

        if abs == 0:
            self.delta = 2*1/(2.0**self.bits-1.0)
            return LinQuant_.apply(x, abs+1.0, self.delta)

        if self.take_new:
            self.abs = abs
            self.take_new = False
        elif self.training:
            self.abs = 0.9*self.abs + 0.1*abs

        self.delta = 2*self.abs/(2.0**self.bits-1.0)
        return LinQuant_.apply(x, self.abs, self.delta)

class LinQuantExpScale(nn.Module):
    def __init__(self, bits) -> None:
        super(LinQuantExpScale, self).__init__()
        self.bits = bits
        self.register_buffer('abs', torch.zeros(1))
        self.take_new = True

    def forward(self, x, fact=1):
        x = x*fact
        abs = torch.max(torch.abs(x.detach().view(-1)))

        if abs == 0:
            self.delta = 2*1/(2.0**self.bits-1.0)
            return LinQuant_.apply(x, abs+1.0, self.delta)

        if self.take_new:
            self.abs = abs
            self.take_new = False
        elif self.training:
            self.abs = 0.89*self.abs + 0.01*expQuant.apply(self.abs) + 0.1*abs

        self.delta = 2*expQuant.apply(self.abs)/(2.0**self.bits-1.0)
        return LinQuant_.apply(x, expQuant.apply(self.abs), self.delta)

