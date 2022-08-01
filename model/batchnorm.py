from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .quantizer import *
from .utils import*

#############################################################################
#                       FUNCTION                                            #
#############################################################################

def calculate_n_new(weight: torch.Tensor,
                bias: torch.Tensor,
                mean: torch.Tensor,
                var: torch.Tensor,
                out_quant: torch.Tensor,
                rexp: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        n = torch.log2(weight/(out_quant*torch.sqrt(var+1e-5)))+rexp.view(-1)
        return n


def calculate_n_fixed_new(weight: torch.Tensor,
                bias: torch.Tensor,
                mean: torch.Tensor,
                var: torch.Tensor,
                out_quant: torch.Tensor,
                rexp: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        n = torch.log2(weight/(out_quant*torch.sqrt(var+1e-5)))+rexp.view(-1)
        nr = n.max()*torch.ones_like(n)
        return nr

def calculate_t_new(weight: torch.Tensor,
                bias: torch.Tensor,
                mean: torch.Tensor,
                var: torch.Tensor,
                out_quant: torch.Tensor,
                rexp: torch.Tensor,
                n: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        t = (-mean*(weight/(out_quant*torch.sqrt(var+1e-5))) + bias/out_quant)
        return t

def calculate_alpha_new(weight: torch.Tensor,
                    mean: torch.Tensor,
                    var: torch.Tensor,
                    out_quant: torch.Tensor,
                    rexp: torch.Tensor,
                    ) -> torch.Tensor:
    with torch.no_grad():
        n = torch.log2(weight*rexp.view(-1)/(out_quant*torch.sqrt(var+1e-5)))
        nr = torch.ceil(n)
        alpha = torch.exp2(n-nr)
    return alpha

def calculate_alpha_fixed_new(weight: torch.Tensor,
                    mean: torch.Tensor,
                    var: torch.Tensor,
                    out_quant: torch.Tensor,
                    rexp: torch.Tensor,
                    ) -> torch.Tensor:
    with torch.no_grad():
        n = torch.log2(weight/(out_quant*torch.sqrt(var+1e-5)))+rexp.view(-1)
        nr = n.max()*torch.ones_like(n)
        nr = torch.ceil(nr)
        alpha = torch.exp2(n-nr)
    raise NotImplementedError()
    return alpha

class BatchNorm2dBase_new(torch.nn.BatchNorm2d):
    def __init__(self, num_features, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None, outQuantBits=8, outQuantDyn=False, fixed_n = False):
        super(BatchNorm2dBase_new, self).__init__(num_features, eps, momentum,
                                              affine, track_running_stats, device, dtype)

        self.register_buffer('n',       torch.zeros(num_features))
        self.register_buffer('t',       torch.zeros(num_features))
        self.register_buffer('alpha',  1./np.sqrt(2.)*torch.ones(num_features))

        self.func_n = calculate_n_new
        self.func_t = calculate_t_new
        self.fixed_n = fixed_n
        if fixed_n:
            self.func_a = calculate_alpha_fixed_new
        else:
            self.func_a = calculate_alpha_new

        self.out_quant = LinQuantExpScale(
            outQuantBits, (1, num_features, 1, 1) if outQuantDyn else (-1,), 1,"floor")

        self.register_buffer('weight_sign',             torch.ones(num_features))
        self.outQuantBits = outQuantBits
        self.init = True

    def get_weight_factor(self):
        def ret_fun(rexp):
            self.alpha = self.func_a(weight = self.weight.view(-1).detach(),
                                    mean = self.running_mean.view(-1).detach(),
                                    var = self.running_var.view(-1).detach(),
                                    out_quant = self.out_quant.delta_out.view(-1).detach(),
                                    rexp = rexp.view(-1).detach())
            # print(f"alpha: {self.alpha.view(-1)}")                         
            return self.alpha[:, None, None, None]
        return ret_fun

    def forward(self, input: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x, rexp = input

        self.weight_sign = torch.sign(self.weight).detach()
        
        if self.training:
            if x.dtype == torch.int32:
                x = super().forward(x.type(self.weight.dtype))
            else:
                x = super().forward(x)

            x = self.out_quant(x)
            return x, rexp

        else:
            with torch.no_grad():
                self.n = self.func_n(   weight          = torch.abs(self.weight.view(-1)),
                                        bias            = self.bias.view(-1),
                                        mean            = self.running_mean.view(-1),
                                        var             = self.running_var.view(-1),
                                        out_quant       = self.out_quant.delta_in.view(-1),
                                        rexp            = rexp.view(-1)).detach()

                

                t = self.func_t(        weight          = self.weight.view(-1),
                                        bias            = self.bias.view(-1),
                                        mean            = self.running_mean.view(-1),
                                        var             = self.running_var.view(-1),
                                        out_quant       = self.out_quant.delta_in.view(-1),
                                        rexp            = rexp.view(-1),
                                        n               = self.n.view(-1)).detach()
                    
                # print(self.n)
                self.n = torch.ceil(self.n).detach()


                tmp = self.weight_sign*torch.exp2(self.n.type(torch.float32))

                self.t = t.div(tmp).floor()

                if torch.any(self.n>0):
                    print("BN to big n high inaccuracy",    self.n.view(-1))
                    print("weight:",                        torch.abs(self.weight.view(-1)))
                    print("var:",                           self.running_var.view(-1))
                    print("out_quant",                      self.out_quant.delta_in.view(-1))
                    print("rexp",                           rexp.view(-1))

                self.n = checkNan.apply(self.n,"BN self.n")
                self.t = checkNan.apply(self.t,"BN self.t")
                tmp = checkNan.apply(tmp,"BN tmp")

                x = checkNan.apply(x,"BN pre add")
                xorig_ = x.detach().clone()
               
                x = checkNan.apply(x,"BN post add")
                x = torch.nan_to_num(x)
                x = x + self.t[None, :, None, None]
                x = x.mul(tmp[None, :, None, None])
                if torch.any(torch.isnan(x)):
                    pos=torch.isnan(x).nonzero()
                    print(pos.shape)
                    print(xorig_[pos[0,0],
                                 pos[0,1],
                                 pos[0,2],
                                 pos[0,3]])
                    pass
                x = checkNan.apply(x,"BN post mul")
                x = x.floor().type(torch.int32)
                x = checkNan.apply(x,"BN post floor")
                x = x.clamp(-(2**(self.outQuantBits-1)),
                                    2**(self.outQuantBits-1) - 1)
                x = checkNan.apply(x,"BN post clamp")
                rexp = torch.log2(self.out_quant.delta_out)
                x = checkNan.apply(x,"BN out")
                return x, rexp
