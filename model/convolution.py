import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Union, Tuple

from .quantizer import *
from .utils import *


class Conv2d_(nn.Conv2d):
    def __init__(self, weights, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1, padding: Union[str, _size_2_t] = 0, dilation: _size_2_t = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', device=None, dtype=None) -> None:
        super(Conv2d_, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        self.weight = nn.Parameter(weights.clone(), requires_grad=True)
        self.register_buffer("used_weights", weights.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.used_weights = Round.apply(self.weight)
        self.used_weights = self.used_weights.clamp(-128, 127)
        return self._conv_forward(x, self.used_weights, None)


class Conv2dQuant(nn.Conv2d):
    def __init__(self,  in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1,
                 padding: Union[str, _size_2_t] = 0, dilation: _size_2_t = 1, groups: int = 1,
                 bias: bool = False, padding_mode: str = 'zeros', device=None, dtype=None,
                 weight_quant=None, weight_quant_bits=None, weight_quant_channel_wise=False, weight_quant_args=None, weight_quant_kargs={}) -> None:
        super(Conv2dQuant, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias, padding_mode, device, dtype)

        if weight_quant_args == None:
            quant_qrgs = (8 if weight_quant_bits == None else weight_quant_bits,
                          (-1,) if not weight_quant_channel_wise else (out_channels, 1, 1, 1),
                          0.1
                          )
        else:
            quant_qrgs = weight_quant_args

        if weight_quant == None:
            self.quantw = LinQuant(*quant_qrgs, **weight_quant_kargs)
        else:
            self.quantw = weight_quant

        self.register_buffer('quant_weight', torch.zeros_like(self.weight))

    def convert(self):
        return Conv2d_(self.used_weights, self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias, self.padding_mode, None, None)

    def forward(self, invals: Tuple[torch.Tensor, torch.Tensor], factor=1) -> torch.Tensor:
        input, rexp = invals

        tmp = self.weight.clone()

        orexp = (torch.mean(rexp)).squeeze()
        rexp_diff = rexp.squeeze() - orexp.unsqueeze(-1)
        tmp = tmp*(2**rexp_diff)[None, :, None, None]

        tmp = self.quantw(tmp, factor)

        if not self.training:
            tmp = torch.round(tmp/self.quantw.delta)
            # only nessesary as /delta can have a slight relative error ~1e-6 in calculations
            self.quant_weight = tmp.detach()
        else:
            tmp = tmp/((2**rexp_diff)[None, :, None, None])

        if torch.any(torch.isnan(tmp)):
            print(torch.max(torch.abs(self.weight.view(-1))))
            print(factor)


        out = self._conv_forward(input, tmp, None)
        if torch.any(torch.isnan(out-out.round())):
            print("WTF")
        if not self.training and (out-out.round()).abs().max()!=0:
            print("post convolution not whole number",(out-out.round()).mean())
        return out, orexp


class LinQuant_new_(torch.autograd.Function):
    @staticmethod
    def forward(self, x, abs, delta,rexp_diff,fact):
        with torch.no_grad():
            x = x*(2**rexp_diff)[None, :, None, None]*fact
            self.save_for_backward(x, abs)
            x = x.clamp(-abs, abs)
            x = x.div(delta, rounding_mode="floor").mul(delta)
            x = x/((2**rexp_diff)[None, :, None, None]*fact)
            if torch.any(torch.isnan(x)):
                print("nan in Linquant forward")
            return x
        
    @staticmethod
    def backward(self, grad_output: torch.Tensor):
        with torch.no_grad():
            val = 1
            x, abs = self.saved_tensors
            grad_output = grad_output.masked_fill(torch.logical_and(
                torch.gt(x, val*abs), torch.gt(grad_output, 0)), 0)
            grad_output = grad_output.masked_fill(torch.logical_and(
                torch.le(x, -val*abs), torch.le(grad_output, 0)), 0)
            if torch.any(torch.isnan(grad_output)):
                print("nan in Linquant back")
            return grad_output.detach(), None, None, None, None


class LinQuantWeight(Quant):
    def __init__(self, bits, size=(-1,), mom1=0.1, mom2=0.01) -> None:
        super(LinQuantWeight, self).__init__(size)
        self.bits = bits
        if size == (-1,):
            self.register_buffer('abs', torch.ones(1))
        else:
            self.register_buffer('abs', torch.ones(size))
        self.take_new = True
        self.mom1 = mom1
        self.mom2 = mom2

    def forward(self, x:torch.Tensor,rexp_diff, fact_fun=None):
        with torch.no_grad():
            abs = get_abs(self,x*(2**rexp_diff)[None, :, None, None])
            if torch.any(abs < 1e-6):
                print("weights to small to quantize")
                self.delta = (2*(self.abs/(2.0**self.bits-1.0))).detach()
                return LinQuant_.apply(x*fact, self.abs, self.delta)

            if self.training and self.take_new:
                self.abs = abs.detach()
                self.take_new = False
                # print("new taken")
            elif self.training:
                # print(f" abs diff:  {(abs.view(-1)-self.abs.view(-1)).abs().max()}")
                self.abs = ((1-self.mom1-self.mom2)*self.abs + self.mom1*abs + self.mom2 *
                            (self.abs/(2.0**self.bits-1.0)) * (2.0**self.bits-1.0)).detach()
            # print(f" old delta: {self.delta.view(-1)}")
            self.delta = (2*(self.abs/(2.0**self.bits-1.0))).detach()
            # print(f" new delta: {self.delta.view(-1)}")
            if fact_fun!=None:
                fact = fact_fun(self.delta)
            else:
                fact = 1
            if torch.any(torch.isnan(self.delta)):
                print("nan in weights")
        # print((self.delta).shape)
        return LinQuant_new_.apply(x, self.abs, self.delta,rexp_diff,fact),fact


class Conv2dQuant_new(nn.Conv2d):
    def __init__(self,  in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1,
                 padding: Union[str, _size_2_t] = 0, dilation: _size_2_t = 1, groups: int = 1,
                 bias: bool = False, padding_mode: str = 'zeros', device=None, dtype=None,
                 weight_quant=None, weight_quant_bits=None, weight_quant_channel_wise=False, weight_quant_args=None, weight_quant_kargs={}) -> None:
        super(Conv2dQuant_new, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias, padding_mode, device, dtype)

        if weight_quant_args == None:
            quant_qrgs = (8 if weight_quant_bits == None else weight_quant_bits,
                          (-1,) if not weight_quant_channel_wise else (out_channels, 1, 1, 1),
                          1
                          )
        else:
            quant_qrgs = weight_quant_args

        if weight_quant == None:
            self.quantw = LinQuantWeight(*quant_qrgs, **weight_quant_kargs)
        else:
            self.quantw = weight_quant

        self.register_buffer('quant_weight', torch.zeros_like(self.weight))

    def convert(self):
        return Conv2d_(self.used_weights, self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias, self.padding_mode, None, None)

    def forward(self, invals: Tuple[torch.Tensor, torch.Tensor], factor_fun=None) -> torch.Tensor:
        input, rexp = invals

        orexp = (torch.mean(rexp)).squeeze()
        rexp_diff = rexp.squeeze() - orexp.unsqueeze(-1)

        tmp = self.weight

        tmp,fact = self.quantw(tmp,rexp_diff,factor_fun)
        # tmp = self.weight
        
        if not self.training:
            # fact = 1
            tmp = torch.round(tmp*fact*(2**rexp_diff)[None, :, None, None]/self.quantw.delta)
            tmp = checkNan.apply( tmp, "conv tmp 2")
            # only nessesary as /delta can have a slight relative error ~1e-6 in calculations
            self.quant_weight = tmp.detach()

        if torch.any(torch.isnan(tmp)):
            print(torch.max(torch.abs(self.weight.view(-1))))

        input = checkNan.apply( input, "conv input")
        out = self._conv_forward(input, tmp, None)
        # if torch.any(torch.isnan(out-out.round())):
        #     print("WTF")
        # if not self.training and (out-out.round()).abs().max()!=0:
        #     print("post convolution not whole number",(out-out.round()).mean())
        return out, orexp


class bias_quant(torch.autograd.Function):
    @staticmethod
    def forward(_,x,rexp):
        x = x.mul(2**(-rexp)).floor().div(2**(-rexp))
        return x
    def backward(_,grad):
        return grad, None

class Conv2dQuantFreeStanding_new(Conv2dQuant_new):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1, padding: Union[str, _size_2_t] = 0, dilation: _size_2_t = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', device=None, dtype=None, weight_quant=None, weight_quant_bits=None, weight_quant_channel_wise=False, weight_quant_args=None, weight_quant_kargs={}) -> None:
        super(Conv2dQuantFreeStanding_new,self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype, weight_quant, weight_quant_bits, weight_quant_channel_wise, weight_quant_args, weight_quant_kargs)
        if self.bias!=None:
            self.register_buffer('quant_bias', torch.zeros_like(self.bias))
    def forward(self, invals: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x = super().forward(invals,None)
        rexp = x[1]+torch.log2(self.quantw.delta)
        val = x[0]
        val = checkNan.apply(val,"Conv fs val 1")
        if self.bias!=None:
            bias = bias_quant.apply(self.bias,rexp)
            if self.training:
                val = val.add(bias[None,:,None,None])
                val = checkNan.apply(val,"Conv fs val 3")
                # print("added bias")
                pass
            else:
                # with torch.no_grad():
                self.quant_bias = checkNan.apply(bias.mul(2**(-rexp))).clone()
                val = checkNan.apply(val,"Conv fs val 5")
                val = val+self.quant_bias[None,:,None,None]
                val = checkNan.apply(val,"Conv fs val 4")
        if torch.any(torch.isnan(self.bias)):
            print("conv fs bias is nan")
        if torch.any(torch.isnan(self.quant_bias)):
            print("conv fs quant_bias is nan")
        
        val = checkNan.apply(val,"Conv fs val 2")
        return (val,rexp)