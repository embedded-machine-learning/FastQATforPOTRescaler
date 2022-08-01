import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t, Tensor
from typing import Union, Tuple

from .quantizer import Quant, get_abs, FakeQuant, LinQuantExpScale, LinQuant
from .utils import *

class LinQuantWeight(Quant):
    def __init__(self, bits, size=(-1,),rounding_mode: str = "trunc", quant_int_dtype = torch.int32) -> None:
        super(LinQuantWeight, self).__init__(size,rounding_mode, quant_int_dtype)
        self.bits = bits
        if size == (-1,):
            self.register_buffer('abs', torch.ones(1))
        else:
            self.register_buffer('abs', torch.ones(size))
        self.take_new = True

        assert(self.bits>0)
        self.register_buffer("delta_in_factor", torch.tensor(2./(2.0**self.bits)))
        self.register_buffer("delta_out_factor", torch.tensor(2./(2.0**self.bits-2)))

        self.register_buffer("max", torch.tensor(2**(self.bits-1)-1))
        self.register_buffer("min", torch.tensor(-(2**(self.bits-1)-1)))
        

    def forward(self, x:torch.Tensor,orexp, rexp_diff, fact_fun):
        with torch.no_grad():
            abs = get_abs(self,x*(rexp_diff.view(1,-1,1,1)))
               
            self.abs = abs.detach()
            self.delta_in = self.abs.mul(self.delta_in_factor).detach()
            self.delta_out = self.abs.mul(self.delta_out_factor).detach()
            
            fact = fact_fun(self.delta_out*orexp).view(-1,1,1,1)

        return FakeQuant(   x               = x.clone(),
                            delta_in        = self.delta_in/((rexp_diff.view(1,-1,1,1)*fact)),
                            delta_out       = self.delta_out/((rexp_diff.view(1,-1,1,1)*fact)),
                            training        = self.training,
                            min_quant       = self.min,
                            max_quant       = self.max,
                            rounding_mode   = self.rounding_mode,
                            quant_int_dtype = self.quant_int_dtype
                        ), fact

class Conv2dQuant_new(nn.Conv2d):
    def __init__(self,  in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1,
                 padding: Union[str, _size_2_t] = 0, dilation: _size_2_t = 1, groups: int = 1,
                 bias: bool = True, padding_mode: str = 'zeros', device=None, dtype=None,
                 weight_quant=None, weight_quant_bits=8,weight_quant_channel_wise=False, weight_quant_args=None, weight_quant_kargs={},
                 out_quant=None,    out_quant_bits=8,   out_quant_channel_wise=False,    out_quant_args=None,    out_quant_kargs={},
                 quant_int_dtype = torch.int32, quant_float_dtype = torch.float32 ) -> None:
        super(Conv2dQuant_new, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias, padding_mode, device, dtype)

        # Weight Quant
        if weight_quant_args == None:
            quant_qrgs = (  weight_quant_bits,
                            (-1,) if not weight_quant_channel_wise else (out_channels, 1, 1, 1),
                            "trunc"
                          )
        else:
            quant_qrgs = weight_quant_args

        if weight_quant == None:
            self.quantw = LinQuantWeight(*quant_qrgs, **weight_quant_kargs)
        else:
            self.quantw = weight_quant

        # Out Quant
        # only used if factor_fun in forward is None
        if out_quant_args == None:
            out_quant_args =    (   out_quant_bits,
                                    (-1,) if not out_quant_channel_wise else (1,out_channels,1,1),
                                    1,
                                    "floor"
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

        self.quant_int_dtype = quant_int_dtype
        self.quant_float_dtype = quant_float_dtype

        self.debug_fact = []
        self.debug_weight = []
        self.debug_bias = []
        self.debug_n = []       


    def get_weight_factor(self,delta_O):
        def fun(rexp):
            with torch.no_grad():
                # print(delta_I,delta_O,delta_W)
                n = rexp.view(-1)/delta_O.view(-1)
                n = torch.log2(n)
                nr = torch.ceil(n)
                return torch.exp2(n-nr)
        return fun
    
    def calculate_n(self,delta_W,delta_I,delta_O):
        with torch.no_grad():
            n = delta_W.view(-1)*delta_I.view(-1)/delta_O.view(-1)
            n = torch.log2(n)
            self.debug_n = n.clone()
            nr = torch.ceil(n)
        return nr

    def forward(self, invals: Tuple[torch.Tensor, torch.Tensor], factor_fun=None) -> torch.Tensor:
        input, rexp = invals

        orexp = (torch.mean(rexp)).squeeze()
        rexp_diff = rexp.squeeze() - orexp.unsqueeze(-1)

        # print("orexp", orexp)
        # print("rexp_diff", rexp_diff.view(-1))

        weight = self.weight

        if factor_fun==None:
            weight,fact = self.quantw(  weight.type(self.quant_float_dtype),
                                        orexp.type(self.quant_float_dtype).exp2(),
                                        rexp_diff.type(self.quant_float_dtype).exp2(),
                                        self.get_weight_factor(self.out_quant.delta_in.view(-1).detach())
                                    )
        else:
            weight,fact = self.quantw(  weight.type(self.quant_float_dtype),
                                        orexp.type(self.quant_float_dtype).exp2(),
                                        rexp_diff.type(self.quant_float_dtype).exp2(),
                                        factor_fun)
        # weight = self.weight

        weight = weight.type(self.weight.dtype)
        fact = fact.type(self.weight.dtype)

        if self.bias == None: 
            bias = None
        else:
            bias = FakeQuant(   x               = self.bias.clone(),
                                delta_in        = self.out_quant.delta_in,
                                delta_out       = self.out_quant.delta_out,
                                training        = self.training,
                                min_quant       = self.out_quant.min,
                                max_quant       = self.out_quant.max,
                                rounding_mode   = self.out_quant.rounding_mode,
                                quant_int_dtype = self.out_quant.quant_int_dtype)
            # print(bias.view(-1),self.bias.view(-1))
            # bias = self.bias.clone() 
        
        self.debug_fact = fact
        self.debug_weight = weight
        self.debug_bias = bias

        if not self.training:
            if bias!=None:
                self.t = bias.detach().view(1,-1,1,1)
            else:
                self.t = None
            # print(weight)
            # only nessesary as /delta can have a slight relative error ~1e-6 in calculations
            self.quant_weight = weight.detach()
            self.n = self.calculate_n(self.quantw.delta_out.view(-1).detach(),2**orexp.view(-1).detach(),self.out_quant.delta_in.view(-1).detach()).view(1,-1,1,1)
            

        # if torch.any(torch.isnan(weight)):
        #     print(torch.max(torch.abs(self.weight.view(-1))))

        # input = checkNan.apply( input, "conv input")
        if self.training:
            out = self._conv_forward(input, weight, bias)
        else:

            out = self._conv_forward(input.type(self.quant_float_dtype), weight.type(self.quant_float_dtype), None).type(self.quant_int_dtype)
  
        
        if factor_fun == None:
            if self.training:
                out2 = self.out_quant(out)
                # print(torch.log2(self.out_quant.delta_out.detach()).view(-1))
            else:
                # print(self.n.shape)
                if bias!=None:
                    out2 = out.mul(torch.exp2(self.n)).add_(self.t).clamp_(self.out_quant.min,self.out_quant.max).floor_()
                else:
                    out2 = out.mul(torch.exp2(self.n)).clamp_(self.out_quant.min,self.out_quant.max).floor_()
            return out2, torch.log2(self.out_quant.delta_out.detach())
        else:
            return out, orexp+self.quantw.delta_out.log2()
