from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .quantizer import *
from .utils import*
#########################################################################################
#                                   FUNCTIONS                                           #
#########################################################################################


def calculate_n(weight: torch.Tensor,
                bias: torch.Tensor,
                mean: torch.Tensor,
                var: torch.Tensor,
                in_quant: torch.Tensor,
                out_quant: torch.Tensor,
                rexp: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        n = torch.log2(in_quant/out_quant*weight/torch.sqrt(var+1e-5))
        nr = torch.round(n+rexp.view(-1))
        # print(n+rexp.view(-1))
        # nr = n+rexp.view(-1)
        return nr.detach()


def calculate_n_fixed(weight: torch.Tensor,
                      bias: torch.Tensor,
                      mean: torch.Tensor,
                      var: torch.Tensor,
                      in_quant: torch.Tensor,
                      out_quant: torch.Tensor,
                      rexp: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        n = torch.log2(in_quant/out_quant*weight/torch.sqrt(var+1e-5))
        nr = torch.round(n+rexp.view(-1))
        nr = torch.median(nr)*torch.ones_like(nr)
        return nr.detach()


def calculate_t(weight: torch.Tensor,
                bias: torch.Tensor,
                mean: torch.Tensor,
                var: torch.Tensor,
                in_quant: torch.Tensor,
                out_quant: torch.Tensor,
                rexp: torch.Tensor,
                n: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        t = torch.round(-mean*(torch.exp2(n-rexp))/in_quant + bias/out_quant)
        return t.detach()


def calculate_alpha(weight: torch.Tensor,
                    bias: torch.Tensor,
                    mean: torch.Tensor,
                    var: torch.Tensor,
                    in_quant: torch.Tensor,
                    out_quant: torch.Tensor,
                    rexp: torch.Tensor,
                    n: torch.Tensor,
                    alpha_old: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        # mom = 0.0
        mom = 0.99 
        mom = 0.9 
        var_new = (weight*in_quant/out_quant).square() * torch.exp2(-2*(n-rexp))-1e-5
        # var_new = (weight*in_quant/out_quant) * torch.exp2(-1*(n-rexp))-1e-5
        alpha = torch.sqrt(var_new/var)
        # print("weight,in_quant,out_quant,n,rexp")
        # print(weight,in_quant,out_quant,n,rexp)
        # print("var,var_new,alpha,alpha_old")
        # print(var,var_new,alpha,alpha_old)
        alpha = alpha.masked_fill(torch.isnan(alpha), 1)
        alpha = mom*alpha_old + (1-mom)*alpha*alpha_old
        cond1 = alpha < 1.05
        cond2 = alpha > 0.3
        alpha = torch.where(cond1, alpha, alpha/2)
        alpha = torch.where(cond2, alpha, alpha*2)
        alpha = alpha.clamp(0.2, 1.06)

        var = var*torch.square(alpha/alpha_old)
        mean = mean*alpha/alpha_old
    return alpha.detach(), var.detach(), mean.detach()


def calculate_alpha_fixed(weight: torch.Tensor,
                          bias: torch.Tensor,
                          mean: torch.Tensor,
                          var: torch.Tensor,
                          in_quant: torch.Tensor,
                          out_quant: torch.Tensor,
                          rexp: torch.Tensor,
                          n: torch.Tensor,
                          alpha_old: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        mom = 0.99
        var_new = (weight*in_quant/out_quant).square() * \
            torch.exp2(-2*(n-rexp))-1e-5
        alpha = torch.sqrt(var_new/var)
        alpha = alpha.masked_fill(torch.isnan(alpha), 1)
        alpha = mom*alpha_old + (1-mom)*alpha*alpha_old
        if torch.any(alpha > 1.05):
            alpha = alpha/2.0
        if torch.all(alpha < 0.4):
            alpha = alpha*2.0
        alpha = alpha.clamp(0, 1.1)

        var = var*torch.square(alpha/alpha_old)
        mean = mean*alpha/alpha_old
    return alpha.detach(), var.detach(), mean.detach()


class BatchNorm2dBasefn(torch.autograd.Function):
    @staticmethod
    def forward(_, self, xorig, other_res):
        # print(self.state_dict())
        with torch.no_grad():
            mean = xorig.mean([0, 2, 3])
            var = xorig.var([0, 2, 3], unbiased=False)
            self.n = self.func_n(weight=torch.abs(self.weight.view(-1)),
                                 bias=self.bias.view(-1),
                                 mean=mean.view(-1),
                                 var=var.view(-1),
                                 in_quant=self.in_quant.view(-1),
                                 out_quant=self.out_quant.delta.view(-1),
                                 rexp=self.rexp.view(-1)).detach()
            self.t = self.func_t(weight=self.weight.view(-1),
                                 bias=self.bias.view(-1),
                                 mean=mean.view(-1),
                                 var=var.view(-1),
                                 in_quant=self.in_quant.view(-1),
                                 out_quant=self.out_quant.delta.view(-1),
                                 rexp=self.rexp.view(-1),
                                 n=self.n.view(-1)).detach()
            self.t = self.t.clamp_(-(2**(self.outQuantBits-1)),
                                   2**(self.outQuantBits-1) - 1).detach()
            tmp = torch.exp2(self.n-self.rexp.view(-1))/self.in_quant.view(-1)
            xorig = xorig.mul_(tmp[None, :, None, None]).add_(self.t[None, :, None, None])
            xorig = xorig.floor_()
            xorig = xorig.clamp_(-(2**(self.outQuantBits-1)),
                                 2**(self.outQuantBits-1) - 1)
            xorig = xorig.mul_(self.out_quant.delta)
            rexp = torch.log2(self.out_quant.delta)
            
            self.alpha, self.running_var, self.running_mean = self.func_a(weight=self.weight.detach().view(-1),
                                                                          bias=self.bias.detach().view(
                                                                              -1),
                                                                          mean=mean.detach().view(
                                                                              -1),
                                                                          var=var.detach().view(
                                                                              -1),
                                                                          in_quant=self.in_quant.detach().view(
                                                                              -1),
                                                                          out_quant=self.out_quant.delta.detach().view(
                                                                              -1),
                                                                          rexp=self.rexp.detach().view(
                                                                              -1),
                                                                          n=self.n.detach().view(
                                                                              -1),
                                                                          alpha_old=self.alpha.detach().view(-1))
        return xorig, rexp

    @staticmethod
    def backward(self, backprob, rexpback):
        return None, None, backprob


#########################################################################################
#                                   CLASSES                                             #
#########################################################################################
class BatchNorm2d_(torch.nn.Module):
    def __init__(self, num_features, n: torch.Tensor, t: torch.Tensor, outQuantBits) -> None:
        super(BatchNorm2d_, self).__init__()
        self.num_features = num_features
        self.outQuantBits = outQuantBits
        self.register_parameter("bias", nn.Parameter(
            t.clone().float(), requires_grad=True))
        self.register_buffer("t", t.clone())
        self.register_buffer("n", n.clone().requires_grad_(False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.t = Round.apply(
            self.bias.clamp(-2**(self.outQuantBits-1), 2**(self.outQuantBits-1)-1))
        x = x * torch.exp2(self.n)[None, :, None, None] + \
            self.t[None, :, None, None]
        x = x.clamp(-2**(self.outQuantBits-1), 2**(self.outQuantBits-1)-1)
        x = Floor.apply(x)
        return x


class BatchNorm2dBase(torch.nn.BatchNorm2d):
    def __init__(self, num_features, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None, outQuantBits=8, outQuantDyn=False):
        super(BatchNorm2dBase, self).__init__(num_features, eps, momentum,
                                              affine, track_running_stats, device, dtype)

        self.register_buffer('n',       torch.zeros(num_features))
        self.register_buffer('t',       torch.zeros(num_features))
        self.register_buffer('alpha',  1./np.sqrt(2.)*torch.ones(num_features))

        self.func_n = calculate_n
        self.func_t = calculate_t
        self.func_a = calculate_alpha

        self.out_quant = LinQuant(
            outQuantBits, (1, num_features, 1, 1) if outQuantDyn else (-1,), 0.1, 0)

        self.register_buffer('in_quant',    torch.ones(num_features, 1, 1, 1))
        self.register_buffer('rexp',        torch.tensor(0.))
        self.register_buffer('weight_sign', torch.ones(num_features))
        self.outQuantBits = outQuantBits
        self.init = True

    def get_weight_factor(self):
        return self.alpha[:, None, None, None]

    def forward(self, input: Tuple[torch.Tensor, torch.Tensor], in_quant=None) -> Tuple[torch.Tensor, torch.Tensor]:
        x, rexp = input
        if in_quant is not None:
            self.in_quant = in_quant.detach()
        self.rexp = rexp.detach()
        xorig = x.detach().clone()
        # if self.training:
        x = super().forward(x)
        # else:
        #     x = super().forward(x*self.in_quant.view(-1)[None,:,None,None])
        self.weight_sign = torch.sign(self.weight).detach()
        if self.training:
            x = self.out_quant(x)
            out = BatchNorm2dBasefn.apply(self, xorig, x)
            
            return out
        else:
            with torch.no_grad():
                self.n = self.func_n(weight=torch.abs(self.weight.view(-1)),
                                    bias=self.bias.view(-1),
                                    mean=self.running_mean.view(-1),
                                    var=self.running_var.view(-1),
                                    in_quant=self.in_quant.view(-1),
                                    out_quant=self.out_quant.delta.view(-1),
                                    rexp=self.rexp.view(-1)).detach()
                self.t = self.func_t(weight=self.weight.view(-1),
                                    bias=self.bias.view(-1),
                                    mean=self.running_mean.view(-1),
                                    var=self.running_var.view(-1),
                                    in_quant=self.in_quant.view(-1),
                                    out_quant=self.out_quant.delta.view(-1),
                                    rexp=self.rexp.view(-1),
                                    n=self.n.view(-1)).detach()
                self.t = self.t.clamp_(-(2**(self.outQuantBits-1)),
                                    2**(self.outQuantBits-1) - 1).detach()
                tmp = torch.exp2(self.n)
                xorig = xorig.mul_(tmp[None, :, None, None]).add_(self.t[None, :, None, None])
                xorig = xorig.floor_()
                xorig = xorig.clamp_(-(2**(self.outQuantBits-1)),
                                    2**(self.outQuantBits-1) - 1)
                rexp = torch.log2(self.out_quant.delta)

                return xorig, rexp


class BatchNorm2dBase_fixed(BatchNorm2dBase):
    def __init__(self, num_features, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None, outQuantBits=8, outQuantDyn=False):
        super(BatchNorm2dBase_fixed, self).__init__(num_features, eps, momentum,
                                                    affine, track_running_stats, device, dtype, outQuantBits, outQuantDyn)

        self.func_n = calculate_n_fixed


#############################################################################
#                       NEW SHIT                                            #
#############################################################################

def calculate_n_new(weight: torch.Tensor,
                bias: torch.Tensor,
                mean: torch.Tensor,
                var: torch.Tensor,
                in_quant: torch.Tensor,
                out_quant: torch.Tensor,
                rexp: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        n = torch.log2(in_quant/out_quant*weight/torch.sqrt(var+1e-5))+rexp.view(-1)
        return n

def calculate_t_new(weight: torch.Tensor,
                bias: torch.Tensor,
                mean: torch.Tensor,
                var: torch.Tensor,
                in_quant: torch.Tensor,
                out_quant: torch.Tensor,
                rexp: torch.Tensor,
                n: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        t = (-mean*(weight/(out_quant*torch.sqrt(var+1e-5))) + bias/out_quant)
        return t

def calculate_alpha_new(weight: torch.Tensor,
                    mean: torch.Tensor,
                    var: torch.Tensor,
                    in_quant: torch.Tensor,
                    out_quant: torch.Tensor,
                    rexp: torch.Tensor,
                    ) -> torch.Tensor:
    with torch.no_grad():
        n = torch.log2(in_quant*weight/(out_quant*torch.sqrt(var+1e-5)))+rexp.view(-1)
        nr = torch.ceil(n)
        alpha = torch.exp2(n-nr)
    return alpha

class BatchNorm2dBase_new(torch.nn.BatchNorm2d):
    def __init__(self, num_features, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None, outQuantBits=8, outQuantDyn=False):
        super(BatchNorm2dBase_new, self).__init__(num_features, eps, momentum,
                                              affine, track_running_stats, device, dtype)

        self.register_buffer('n',       torch.zeros(num_features))
        self.register_buffer('t',       torch.zeros(num_features))
        self.register_buffer('alpha',  1./np.sqrt(2.)*torch.ones(num_features))

        self.func_n = calculate_n_new
        self.func_t = calculate_t_new
        self.func_a = calculate_alpha_new

        self.out_quant = LinQuant(
            outQuantBits, (1, num_features, 1, 1) if outQuantDyn else (-1,), 0.1, 0)

        self.register_buffer('in_quant',    torch.ones(num_features, 1, 1, 1))
        self.register_buffer('rexp',        torch.tensor(0.))
        self.register_buffer('weight_sign', torch.ones(num_features))
        self.outQuantBits = outQuantBits
        self.init = True

    def get_weight_factor(self):
        def ret_fun(in_quant):
            self.in_quant = in_quant.detach()
            self.alpha = self.func_a(weight = self.weight.view(-1).detach(),
                                    mean = self.running_mean.view(-1).detach(),
                                    var = self.running_var.view(-1).detach(),
                                    in_quant = self.in_quant.view(-1).detach(),
                                    out_quant = self.out_quant.delta.view(-1).detach(),
                                    rexp = self.rexp.view(-1).detach())
            # print(f"alpha: {self.alpha.view(-1)}")                         
            return self.alpha[:, None, None, None]
        return ret_fun

    def forward(self, input: Tuple[torch.Tensor, torch.Tensor], in_quant=None) -> Tuple[torch.Tensor, torch.Tensor]:
        x, rexp = input
        if in_quant is not None:
            self.in_quant = in_quant.detach()
        self.rexp = rexp.detach()
        xorig = x.detach().clone()
        # if self.training:
        x = super().forward(x)
        # else:
        #     x = super().forward(x*self.in_quant.view(-1)[None,:,None,None])
        self.weight_sign = torch.sign(self.weight).detach()
        if self.training:
            x = self.out_quant(x)
            with torch.no_grad():
                mean = xorig.mean([0, 2, 3])
                var = xorig.var([0, 2, 3], unbiased=False)
                # print(f"var : {var.view(-1)} and {self.running_var.view(-1)}")
                # print(f"mean : {mean.view(-1)} and {self.running_mean.view(-1)}")
                self.n = self.func_n(weight=torch.abs(self.weight.view(-1)),
                                    bias=self.bias.view(-1),
                                    mean=mean.view(-1),
                                    var=var.view(-1),
                                    in_quant=self.in_quant.view(-1),
                                    out_quant=self.out_quant.delta.view(-1),
                                    rexp=self.rexp.view(-1)).detach()
                self.t = self.func_t(weight=self.weight.view(-1),
                                    bias=self.bias.view(-1),
                                    mean=mean.view(-1),
                                    var=var.view(-1),
                                    in_quant=self.in_quant.view(-1),
                                    out_quant=self.out_quant.delta.view(-1),
                                    rexp=self.rexp.view(-1),
                                    n=self.n.view(-1)).detach()

                if torch.any(self.n>0):
                    print("BN to big n high inaccuracy", self.n.view(-1))

                # self.t = self.t.clamp(-(2**(self.outQuantBits-1)),
                #                     2**(self.outQuantBits-1) - 1).detach()
                tmp = self.weight_sign*torch.exp2(self.n-self.rexp.view(-1))/self.in_quant.view(-1)
                self.t = self.t.div(tmp)
                xorig = xorig.add(self.t[None, :, None, None]).mul(tmp[None, :, None, None])
                xorig = xorig.floor()
                xorig = xorig.clamp(-(2**(self.outQuantBits-1)),
                                    2**(self.outQuantBits-1) - 1)
                xorig = xorig.mul(self.out_quant.delta)
                rexp = torch.log2(self.out_quant.delta)

                x.data = xorig.detach()
            return x, rexp

        else:
            with torch.no_grad():
                self.n = self.func_n(weight=torch.abs(self.weight.view(-1)),
                                    bias=self.bias.view(-1),
                                    mean=self.running_mean.view(-1),
                                    var=self.running_var.view(-1),
                                    in_quant=self.in_quant.view(-1),
                                    out_quant=self.out_quant.delta.view(-1),
                                    rexp=self.rexp.view(-1)).detach()

                

                self.t = self.func_t(weight=self.weight.view(-1),
                                    bias=self.bias.view(-1),
                                    mean=self.running_mean.view(-1),
                                    var=self.running_var.view(-1),
                                    in_quant=self.in_quant.view(-1),
                                    out_quant=self.out_quant.delta.view(-1),
                                    rexp=self.rexp.view(-1),
                                    n=self.n.view(-1)).detach()
                
                # print(f"n : {self.n.view(-1)}")
                self.n = torch.ceil(self.n)
                if torch.any(self.n>0):
                    print("BN to big n high inaccuracy", self.n.view(-1))
                # print(f"n : {self.n.view(-1)}")

                # # self.n = torch.ceil(self.n)
                # self.t = self.t.clamp_(-(2**(self.outQuantBits-1)),
                #                     2**(self.outQuantBits-1) - 1).detach()
                tmp = self.weight_sign*torch.exp2(self.n)
                self.t = self.t.div(tmp).round()
                xorig = xorig.add(self.t[None, :, None, None]).mul(tmp[None, :, None, None])
                # xorig = xorig.mul_(tmp[None, :, None, None]).add(self.t[None, :, None, None])
                xorig = xorig.floor()
                xorig = xorig.clamp(-(2**(self.outQuantBits-1)),
                                    2**(self.outQuantBits-1) - 1)
                rexp = torch.log2(self.out_quant.delta)

                return xorig, rexp
