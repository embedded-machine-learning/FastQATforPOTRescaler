from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.quantizer import *
from model.utils import*


class BatchNorm2dQuantFixed(nn.Module): 
    def __init__(self, num_features, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(BatchNorm2dQuantFixed, self).__init__()

        self.weight = torch.nn.Parameter(
            torch.empty(num_features, **factory_kwargs))
        self.bias = torch.nn.Parameter(
            torch.empty(num_features, **factory_kwargs))
        torch.nn.init.ones_(self.weight)
        torch.nn.init.zeros_(self.bias)
        self.register_buffer('n', torch.zeros(num_features))
        self.register_buffer('t', torch.zeros(num_features))

        # running values
        self.register_buffer('sig', torch.ones(num_features))
        self.register_buffer('mu', torch.zeros(num_features))

        self.first = True

        # from quant
        self.quant = LinQuant(8,(-1,),0.1,0)
        self.register_buffer('inference_n', torch.ones(num_features))

        # for weights
        self.register_buffer('alpha', torch.ones(num_features))

    def calculate_n(self,sig,gamma,delta_in,delta_out,rexp) -> Tuple[torch.Tensor,torch.Tensor]:#
        n = delta_in.view(-1)/delta_out.view(-1)*gamma.view(-1)/torch.sqrt(sig.view(-1)+1e-5)
        nr = torch.median(torch.round(torch.log2(n)))*torch.ones_like(torch.round(torch.log2(n)))
        return nr,nr+rexp.view(-1)
    def calculate_t(self,sig,mu,gamma,beta,delta_out) -> torch.Tensor:
        t =  -mu.view(-1)*(gamma.view(-1))/(torch.sqrt(sig.view(-1)+1e-5) * delta_out.view(-1)) + beta.view(-1)/delta_out.view(-1)
        t = torch.round(t).clamp(-128, 127)
        return t

    def calculate_alpha(self,delta_in)-> None:
         with torch.no_grad():
            mom = 0.99
            sig = (self.weight.view(-1)*delta_in.view(-1)/self.quant.delta.view(-1)).square() * torch.exp2(-2*self.n.view(-1))-1e-5
            alpha = torch.sqrt(sig.view(-1)/self.sig.view(-1))
            alpha = alpha.masked_fill(torch.isnan(alpha), 1)
            old_alpha = self.alpha
            self.alpha = mom*self.alpha + (1-mom)*alpha*self.alpha
            if torch.any(self.alpha>1):
                self.alpha = self.alpha/2.0
            self.sig = self.sig*torch.square(self.alpha/old_alpha)
            self.mu = self.mu*self.alpha/old_alpha


    def forward(self,  invals: Tuple[torch.Tensor, torch.Tensor], in_quant=1):
        x, rexp = invals
        if self.training:
            mu = x.mean([0, 2, 3])
            sig = x.var([0, 2, 3], unbiased=False)
            with torch.no_grad():
                mom = 0.95
                self.mu = self.mu*mom + (1-mom)*mu.squeeze()
                self.sig = self.sig*mom + (1-mom)*sig.squeeze()
                # self.old_sigs.append(self.sig.cpu())
                # self.true_old_suigs.append(sig.cpu())
                # self.old_alpha.append(self.alpha.cpu())
                if torch.any(torch.isnan(self.sig)):
                    self.sig = sig
                if torch.any(torch.isnan(self.mu)):
                    self.mu = mu
                if self.first:
                    self.first = False
                    self.sig = sig
                    self.mu = mu
            

            xorig = x.clone()
            # clamp to min 0 so n can't be negative
            weights_used = self.weight.clamp(0)

            x = (x-mu[None, :, None, None]) / (torch.sqrt(sig[None, :, None, None]+1e-5))
            x = x*weights_used[None, :, None, None] + self.bias[None, :, None, None]

            x = self.quant(x)
            x = (x)/self.quant.delta

            with torch.no_grad():
                self.n, self.inference_n = self.calculate_n(sig=sig,gamma=weights_used,delta_in=in_quant,delta_out=self.quant.delta,rexp=rexp)
                self.t = self.calculate_t(sig,mu,weights_used,self.bias,self.quant.delta)
                xorig = xorig * torch.exp2(self.n)[None, :, None, None]/in_quant.view(-1)[None, :, None, None] + self.t[None, :, None, None]
                xorig = torch.round(xorig)
                xorig = torch.clamp(xorig, -128, 127)

            x, xorig = switch.apply(x, xorig)
            tmp = torch.round(torch.log2(self.quant.delta))


            rexp=-6.0*torch.ones_like(rexp)

            x = x*(2**rexp[None,:,None,None])


            # x = x/2**6
            # x = x*torch.exp2(tmp)
            self.calculate_alpha(in_quant)
            return x, rexp
        else:
            with torch.no_grad():
                mu = self.mu
                sig = self.sig
                # clamp to min 0 so n can't be negative
                weights_used = self.weight.clamp(0)

                self.n, self.inference_n = self.calculate_n(sig=sig,gamma=weights_used,delta_in=in_quant,delta_out=self.quant.delta,rexp=rexp)
                self.t = self.calculate_t(sig,mu,weights_used,self.bias,self.quant.delta)
               
                x = x*torch.exp2(self.inference_n)[None, :, None, None] + self.t[None, :, None, None]
                x = torch.round(x)
                x = torch.clamp(x, -128, 127)

                rexp=-6*torch.ones_like(rexp)
            # running_exp=tmp = torch.round(torch.log2(self.quant.desired_delta))
            return x, rexp

        
    def get_weight_factor(self):
        return self.alpha[:, None, None, None]


class BatchNorm2dQuantFixedDynOut(BatchNorm2dQuantFixed):
    def __init__(self, num_features, device=None, dtype=None):
        super(BatchNorm2dQuantFixedDynOut,self).__init__(num_features, device, dtype)
        self.quant = LinQuantExpScale(8,(-1,),0.1,0.01)

    def calculate_t(self,mu,n,beta,delta_out,delta_in) -> torch.Tensor:
        t =  -mu.view(-1)*(2**n.view(-1))/delta_in.view(-1) + beta.view(-1)/delta_out.view(-1)
        t = torch.round(t).clamp(-128, 127)
        return t

    def forward(self,  invals: Tuple[torch.Tensor, torch.Tensor], in_quant=1):
        x, rexp = invals
        if self.training:
            mu = x.mean([0, 2, 3])
            sig = x.var([0, 2, 3], unbiased=False)
            with torch.no_grad():
                mom = 0.95
                self.mu = self.mu*mom + (1-mom)*mu.squeeze()
                self.sig = self.sig*mom + (1-mom)*sig.squeeze()
                # self.old_sigs.append(self.sig.cpu())
                # self.true_old_suigs.append(sig.cpu())
                # self.old_alpha.append(self.alpha.cpu())
                if torch.any(torch.isnan(self.sig)):
                    self.sig = sig
                if torch.any(torch.isnan(self.mu)):
                    self.mu = mu
                if self.first:
                    self.first = False
                    self.sig = sig
                    self.mu = mu
            

            xorig = x.clone()
            # clamp to min 0 so n can't be negative
            weights_used = self.weight.clamp(0)

            x = (x-mu[None, :, None, None]) / (torch.sqrt(sig[None, :, None, None]+1e-5))
            x = x*weights_used[None, :, None, None] + self.bias[None, :, None, None]

            x = self.quant(x)
            # x = (x)/self.quant.delta.detach()

            with torch.no_grad():
                self.n, self.inference_n = self.calculate_n(sig=sig,gamma=weights_used,delta_in=in_quant,delta_out=self.quant.delta,rexp=rexp)
                
                self.t = self.calculate_t(mu=mu,n=self.n,beta=self.bias,delta_out=self.quant.delta,delta_in=in_quant)
                xorig = xorig * torch.exp2(self.n)[None, :, None, None]/in_quant.view(-1)[None, :, None, None] + self.t[None, :, None, None]
                xorig = torch.round(xorig)
                xorig = torch.clamp(xorig, -128, 127)
                tmp = -mu*(weights_used)/(torch.sqrt(sig+1e-5) * self.quant.delta) + self.bias/self.quant.delta
                rexp=torch.log2(self.quant.delta)
                xorig = xorig*(2**rexp[None,:,None,None])


            x, xorig = switch.apply(x, xorig)
            tmp = torch.round(torch.log2(self.quant.delta))

           

            # x = x/2**6
            # x = x*torch.exp2(tmp)
            self.calculate_alpha(in_quant)
            return x, rexp
        else:
            with torch.no_grad():
                mu = self.mu
                sig = self.sig
                # clamp to min 0 so n can't be negative
                weights_used = self.weight.clamp(0)

                self.n, self.inference_n = self.calculate_n(sig=sig,gamma=weights_used,delta_in=in_quant,delta_out=self.quant.delta,rexp=rexp)
                self.t = self.calculate_t(mu=mu,n=self.n,beta=self.bias,delta_out=self.quant.delta,delta_in=in_quant)

                x = x*torch.exp2(self.inference_n)[None, :, None, None] + self.t[None, :, None, None]
                x = torch.round(x)
                x = torch.clamp(x, -128, 127)

                rexp=torch.log2(self.quant.delta)
            # running_exp=tmp = torch.round(torch.log2(self.quant.desired_delta))
            return x, rexp