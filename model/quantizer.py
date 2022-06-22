import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import *


#################################################
#           FUNCTIONS                           #
#################################################


class expQuant(torch.autograd.Function):
    @staticmethod
    def forward(self, x):
        tmp = torch.exp2(torch.floor(torch.log2(x)))
        if torch.any(torch.isnan(tmp)):
            print("nan in exp scale")
        return tmp

    @staticmethod
    def backward(self, grad_output):
        if torch.any(torch.isnan(grad_output)):
            print("nan in exp scale")
        return grad_output.detach()


class LinQuant_(torch.autograd.Function):
    @staticmethod
    def forward(self, x, abs, delta):
        with torch.no_grad():
            self.save_for_backward(x, abs)
            x = x.clamp_(-abs, abs)
            x = x.div_(delta,rounding_mode="floor").mul_(delta)
            if torch.any(torch.isnan(x)):     
                print("nan in Linquant forward")
            return x
        

    @staticmethod
    def backward(self, grad_output: torch.Tensor):
        with torch.no_grad():
            x, abs = self.saved_tensors
            grad_output = grad_output.masked_fill_(torch.logical_and(
                torch.gt(x, 2*abs), torch.gt(grad_output, 0)), 0)
            grad_output = grad_output.masked_fill_(torch.logical_and(
                torch.le(x, -2*abs), torch.le(grad_output, 0)), 0)
            if torch.any(torch.isnan(grad_output)):
                print("nan in Linquant back")
            return grad_output.detach(), None, None


class specialExpQuant(torch.autograd.Function):
    def forward(self, x: torch.Tensor):
        self.save_for_backward(x)
        x = x.clamp(0, 1)
        cond = torch.gt(x, 2**-6)
        n = torch.where(cond, torch.log2(
            expQuant.apply(x)), torch.zeros_like(x))
        x = torch.where(cond, expQuant.apply(x), torch.zeros_like(x))
        if torch.any(torch.isnan(x)):     
            print("nan in specialExpQuant forward")
        return x.detach(), n.detach(), cond.detach()

    def backward(self, grad_output: torch.Tensor, n_output: torch.Tensor, cond_output: torch.Tensor):
        x, = self.saved_tensors

        # only allow positive graients when greater 1 so that it can still get smaller
        grad_output = grad_output.masked_fill(torch.logical_and(
            torch.gt(x, 2.0), torch.gt(grad_output, 0)), 0)
        # only allow negative graients when smaller 2**-6 so that it can still get bigger
        grad_output = grad_output.masked_fill(torch.logical_and(
            torch.le(x, (2**-7)*1.0), torch.le(grad_output, 0)), 0)
        if torch.any(torch.isnan(grad_output)):     
            print("nan in specialExpQuant backward")
        return grad_output.detach(), None, None


#################################################
#           MODULES                             #
#################################################
class Quant(nn.Module):
    def __init__(self, size) -> None:
        super(Quant, self).__init__()
        if size == (-1,):
            self.register_buffer('delta', torch.zeros(1))
            self.size = (1,)
        else:
            self.register_buffer('delta', torch.zeros(size))
            self.size = size

    def forward(self, x):
        raise NotImplementedError()


class LinQuant(Quant):
    def __init__(self, bits, size=(-1,), mom1=0.1, mom2=0.01) -> None:
        super(LinQuant, self).__init__(size)
        self.bits = bits
        if size == (-1,):
            self.register_buffer('abs', torch.zeros(1))
        else:
            self.register_buffer('abs', torch.zeros(size))
        self.take_new = True
        self.mom1 = mom1
        self.mom2 = mom2

    def forward(self, x, fact=1):
        with torch.no_grad():
            abs = torch.max(torch.abs(x.view(list(
                self.size)+[-1])), dim=(len(self.size)), keepdim=True).values.view(self.size)

            if torch.any(abs < 1e-6):
                print("weights to small to quantize")

            abs = abs.masked_fill(abs < 1e-6, 1e-6)

            if self.training and self.take_new:
                self.abs = abs
                self.take_new = False
            elif self.training:
                self.abs = (1-self.mom1-self.mom2)*self.abs + self.mom1*abs + self.mom2 * (self.abs/(2.0**self.bits-1.0)) * (2.0**self.bits-1.0)

            self.delta = 2*(self.abs/(2.0**self.bits-1.0))
        return LinQuant_.apply(x*fact, self.abs, self.delta)


class LinQuantExpScale(Quant):
    def __init__(self, bits, size=(-1,), mom1=0.1, mom2=0.01) -> None:
        super(LinQuantExpScale, self).__init__(size)
        self.bits = bits
        if size == (-1,):
            self.register_buffer('abs', torch.zeros(1))
        else:
            self.register_buffer('abs', torch.zeros(size))
        self.take_new = True
        self.mom1 = mom1
        self.mom2 = mom2

    def forward(self, x, fact=1):
        with torch.no_grad():
            abs = torch.max(torch.abs(x.view(list(
                self.size)+[-1])), dim=(len(self.size)), keepdim=True).values.view(self.size)

            if torch.any(abs < 1e-6):
                print("weights to small to quantize")

            abs = abs.masked_fill(abs < 1e-6, 1e-6)

            if self.training and self.take_new:
                self.abs = abs
                self.take_new = False
            elif self.training:
                self.abs = (1-self.mom1-self.mom2)*self.abs + self.mom1*abs + self.mom2 * expQuant.apply(self.abs/(2.0**self.bits-1.0)) * (2.0**self.bits-1.0)

            self.delta = 2*expQuant.apply(self.abs/(2.0**self.bits-1.0))
        x = x*fact
        return LinQuant_.apply(x, expQuant.apply(self.abs), self.delta)
