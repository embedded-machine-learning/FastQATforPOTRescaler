from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantizer import Quant, get_abs, LinQuantExpScale, LinQuant, FakeQuant
from .utils     import checkNan





class WeightQuantLinear_(torch.autograd.Function):
    @staticmethod
    def forward(self, x, abs, delta,rexp_diff,fact):
        with torch.no_grad():
            # print("x.shape",x.shape)
            # print("rexp_diff.shape",rexp_diff.shape)
            # print("fact.shape",fact.shape)
            x = (x*(2**rexp_diff.view(1,-1)))*fact.view(-1,1)
            self.save_for_backward(x, abs)
            x = x.clamp(-abs, abs)
            x = x.div(delta, rounding_mode="floor").mul(delta)
            x = x/((2**rexp_diff.view(1,-1))*fact.view(-1,1))
            # print(x.shape)
            if torch.any(torch.isnan(x)):
                print("nan in WeightQuantLinear_ forward")
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
                print("nan in WeightQuantLinear_ back")
            return grad_output.detach(), None, None, None, None


class WeightQuantLinear(Quant):
    def __init__(self, bits, size=(-1,), mom1=0.1, mom2=0.01) -> None:
        super(WeightQuantLinear, self).__init__(size)
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
            abs = get_abs(self,x*(2**rexp_diff)[None, :])
            # print(abs.shape)
            if torch.any(abs < 1e-6):
                print("weights to small to quantize")
                self.delta = (2*(self.abs.type(abs.dtype)/(2.0**self.bits.type(abs.dtype)-1.0))).detach().type(abs.dtype)
                if fact_fun!=None:
                    fact = fact_fun(self.delta)
                else:
                    fact = 1
                if torch.any(torch.isnan(self.delta)):
                    print("nan in WeightQuantLinear weights")
                # print((self.delta).shape)
                return WeightQuantLinear_.apply(x, self.abs, self.delta,rexp_diff,fact),fact

               
            self.abs = abs.detach()
            # print(f" old delta: {self.delta.view(-1)}")
            self.delta = (2*(self.abs/(2.0**self.bits-1.0))).detach()
            # print(f" new delta: {self.delta.view(-1)}")
            if fact_fun!=None:
                fact = fact_fun(self.delta)
            else:
                fact = 1
            if torch.any(torch.isnan(self.delta)) or torch.any(torch.isnan(fact)):
                print("nan in WeightQuantLinear weights")
        # print((self.delta).shape)
        return WeightQuantLinear_.apply(x, self.abs, self.delta,rexp_diff,fact),fact




class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None, 
                weight_quant=None, weight_quant_bits=8, weight_quant_channel_wise=False, weight_quant_args=None, weight_quant_kargs={},
                out_quant = None, out_quant_bits=8, out_quant_channel_wise=False, out_quant_args=None, out_quant_kargs={}) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)

        if weight_quant_args == None:
            quant_qrgs = (weight_quant_bits,
                          (-1,) if not weight_quant_channel_wise else (out_features, 1),
                          1
                          )
        else:
            quant_qrgs = weight_quant_args

        if weight_quant == None:
            self.quantw = WeightQuantLinear(*quant_qrgs, **weight_quant_kargs)
        else:
            self.quantw = weight_quant

        # only used if factor_fun in forward is None
        if out_quant_args == None:
            out_quant_args =    (   out_quant_bits,
                                    (-1,) if not out_quant_channel_wise else (1,out_features),
                                    0.1,
                                    0.01
                                )
        
        if out_quant == None:
            self.out_quant = LinQuantExpScale(*out_quant_args, **out_quant_kargs)
        else:
            self.out_quant = out_quant

        self.register_buffer('quant_weight', torch.zeros_like(self.weight))
        self.register_buffer('n', torch.zeros((out_features if out_quant_channel_wise else 1)))
        if bias:
            self.register_buffer('t', torch.zeros((out_features)))
        else:
            self.t = None
    
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
            weight = torch.round(weight*fact.view(-1,1)*(2**rexp_diff.view(1,-1))/self.quantw.delta.view(-1,1).detach())
            weight = checkNan.apply( weight, "Linear weight 2").type(torch.int32)
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
            self.n = self.calculate_n(self.quantw.delta.view(-1).detach(),2**orexp.view(-1).detach(),self.out_quant.delta.view(-1).detach())
            # print(self.n)
            

        if torch.any(torch.isnan(weight)):
            print(torch.max(torch.abs(self.weight.view(-1))))

        input = checkNan.apply( input, "Linear input")
        if self.training:
            # print("input.shape",input.shape)
            # print("weight.shape",weight.shape)
            # print("bias.shape",bias.shape)
            out = F.linear(input, weight, bias)
        else:
            out = F.linear(input.type(torch.float32), weight.type(torch.float32), bias.type(torch.float32)).type(torch.int32)
        # if torch.any(torch.isnan(out-out.round())):
        #     print("WTF")
        # if not self.training and (out-out.round()).abs().max()!=0:
        #     print("post convolution not whole number",(out-out.round()).mean())

        if factor_fun == None:
            if self.training:
                out = self.out_quant(out)
            else:
                out = torch.floor(out.mul(torch.exp2(self.n))) 
            return out, torch.log2(self.out_quant.delta.detach())
        else:
            return out, orexp