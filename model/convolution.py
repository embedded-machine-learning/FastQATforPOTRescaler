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

class Conv2dQuant_new(nn.Conv2d):
    def __init__(self,  in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1,
                 padding: Union[str, _size_2_t] = 0, dilation: _size_2_t = 1, groups: int = 1,
                 bias: bool = True, padding_mode: str = 'zeros', device=None, dtype=None,
                 weight_quant=None, weight_quant_bits=8,weight_quant_channel_wise=False, weight_quant_args=None, weight_quant_kargs={},
                 out_quant=None,    out_quant_bits=8,   out_quant_channel_wise=False,    out_quant_args=None,    out_quant_kargs={}) -> None:
        super(Conv2dQuant_new, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias, padding_mode, device, dtype)

        if weight_quant_args == None:
            quant_qrgs = (  weight_quant_bits,
                            (-1,) if not weight_quant_channel_wise else (out_channels, 1, 1, 1),
                            1
                          )
        else:
            quant_qrgs = weight_quant_args

        if weight_quant == None:
            self.quantw = LinQuantWeight(*quant_qrgs, **weight_quant_kargs)
        else:
            self.quantw = weight_quant

         # only used if factor_fun in forward is None
        if out_quant_args == None:
            out_quant_args =    (   out_quant_bits,
                                    (-1,) if not out_quant_channel_wise else (1,out_channels,1,1),
                                    0.1,
                                    0.01
                                )
        
        if out_quant == None:
            self.out_quant = LinQuantExpScale(*out_quant_args, **out_quant_kargs)
        else:
            self.out_quant = out_quant

        self.register_buffer('quant_weight', torch.zeros_like(self.weight))
        self.register_buffer('n', torch.zeros((out_channels if out_quant_channel_wise else 1)))
        if bias:
            self.register_buffer('t', torch.zeros((out_channels)))
        else:
            self.t = None

    def convert(self):
        return Conv2d_(self.used_weights, self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias, self.padding_mode, None, None)

    def get_weight_factor(self,delta_I,delta_O):
        def fun(delta_W):
            with torch.no_grad():
                # print(delta_I,delta_O,delta_W)
                n = delta_W.view(-1)*delta_I.view(-1)/delta_O.view(-1)
                n = torch.log2(n)
                nr = torch.ceil(n)
                return torch.exp2(n-nr)
        return fun
    
    def calculate_n(self,delta_W,delta_I,delta_O):
        with torch.no_grad():
            n = delta_W.view(-1)*delta_I.view(-1)/delta_O.view(-1)
            n = torch.log2(n)
            nr = torch.ceil(n)
        return nr

    def forward(self, invals: Tuple[torch.Tensor, torch.Tensor], factor_fun=None) -> torch.Tensor:
        input, rexp = invals

        orexp = (torch.mean(rexp)).squeeze()
        rexp_diff = rexp.squeeze() - orexp.unsqueeze(-1)

        weight = self.weight

        if factor_fun==None:
            weight,fact = self.quantw(weight.type(torch.float32),rexp_diff.type(torch.float32),self.get_weight_factor(orexp.detach().view(-1).exp2(),self.out_quant.delta.view(-1).detach()))
        else:
            weight,fact = self.quantw(weight.type(torch.float32),rexp_diff.type(torch.float32),factor_fun)
        # weight = self.weight
        
        weight = weight.type(self.weight.dtype)
        fact = fact.type(self.weight.dtype)

        if self.bias == None: 
            bias = None
        else:
            bias_fact = (orexp.detach().exp2().view(-1)*self.quantw.delta.view(-1).detach())/fact.view(-1).detach()
            # print(bias_fact)
            bias = FakeQuant.apply(self.bias,bias_fact)


        if not self.training:
            # fact = 1
            weight = torch.round(weight*fact*(2**rexp_diff)[None, :, None, None]/self.quantw.delta)
            weight = checkNan.apply( weight, "conv weight 2").type(torch.int32)
            if bias!=None:
                # print(bias)
                bias = torch.round(bias/bias_fact)
                # print(bias)
                self.t = bias.detach()
            else:
                self.t = None
            # print(weight)
            # only nessesary as /delta can have a slight relative error ~1e-6 in calculations
            self.quant_weight = weight.detach()
            self.n = self.calculate_n(self.quantw.delta.view(-1).detach(),2**orexp.view(-1).detach(),self.out_quant.delta.view(-1).detach()).view(1,-1,1,1)
            

        if torch.any(torch.isnan(weight)):
            print(torch.max(torch.abs(self.weight.view(-1))))

        input = checkNan.apply( input, "conv input")
        if self.training:
            out = self._conv_forward(input, weight, bias)
        else:
            out = self._conv_forward(input.type(torch.float32), weight.type(torch.float32), bias).type(torch.int32)
        # if torch.any(torch.isnan(out-out.round())):
        #     print("WTF")
        # if not self.training and (out-out.round()).abs().max()!=0:
        #     print("post convolution not whole number",(out-out.round()).mean())
        if factor_fun == None:
            if self.training:
                out2 = self.out_quant(out)
                print(torch.log2(self.out_quant.delta.detach()).view(-1))
            else:
                # print(self.n.shape)
                out2 = torch.floor(out.mul(torch.exp2(self.n))) 
            return out2, torch.log2(self.out_quant.delta.detach())
        else:
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