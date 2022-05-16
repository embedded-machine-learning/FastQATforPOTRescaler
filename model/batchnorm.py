from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model_old.quantizer import *
from model_old.utils import*


class BatchNorm2dQuantFixed(nn.Module): 
    def __init__(self, num_features, device=None, dtype=None):
        super(BatchNorm2dQuantFixed, self).__init__()

        self.register_parameter('weight',torch.ones(num_features))
        self.register_parameter('bias',torch.zeros(num_features))
        self.register_buffer('n', torch.zeros(num_features))
        self.register_buffer('t', torch.zeros(num_features))

        # running values
        self.register_buffer('sig', torch.ones(num_features))
        self.register_buffer('mu', torch.zeros(num_features))

        self.first = True

        # from quant
        self.quant = LinQuant(8,(1,num_features,1,1))
        self.register_buffer('inference_n', torch.ones(num_features))

        # for weights
        self.register_buffer('alpha', torch.ones(num_features))

    def calculate_n(self,sig,gamma,delta_in,delta_out,rexp) -> Tuple[torch.Tensor,torch.Tensor]:
        n = delta_in.view(-1)/delta_out.view(-1)*gamma.view(-1)/torch.sqrt(sig+1e-5)
        nr = torch.median(torch.round(torch.log2(n)))*torch.ones_like(torch.round(torch.log2(n)))
        return nr,nr+rexp

    def calculate_t(self,sig,mu,gamma,beta,delta_out) -> torch.Tensor:
        t =  -mu*(gamma)/(torch.sqrt(sig+1e-5) * delta_out) + beta/delta_out
        t = torch.round(t).clamp(-128, 127)
        return t

    def calculate_alpha(self,delta_in)-> None:
         with torch.no_grad():
            mom = 0.99
            sig = (self.weight*delta_in.view(-1)/self.quant.delta).square() * torch.exp2(-2*self.n)-1e-5
            alpha = torch.sqrt(sig/self.sig)
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
                self.n, self.inference_n = self.calculate_n(sig,weights_used,in_quant,self.quant.delta,rexp)
                self.t = self.calculate_t(sig,mu,weights_used,self.bias,self.quant.delta)
                

            xorig = xorig * torch.exp2(self.n)[None, :, None, None]/in_quant.view(-1)[None, :, None, None] + self.t[None, :, None, None]
            xorig = torch.round(xorig)
            xorig = torch.clamp(xorig, -128, 127)

            x, xorig = switch.apply(x, xorig)
            tmp = torch.round(torch.log2(self.quant.delta))

            rexp=-6*torch.ones_like(rexp)

            x = x*(2**rexp)


            # x = x/2**6
            # x = x*torch.exp2(tmp)
            self.calculate_alpha()
            return x, rexp
        else:
            with torch.no_grad():
                mu = self.mu
                sig = self.sig
                # clamp to min 0 so n can't be negative
                weights_used = self.weight.clamp(0)

                self.n, self.inference_n = self.calculate_n(sig,weights_used,in_quant,self.quant.delta,rexp)
                self.t = self.calculate_t(sig,mu,weights_used,self.bias,self.quant.delta)

                x = x*torch.exp2(self.inference_n)[None, :, None, None] + self.t[None, :, None, None]
                x = torch.round(x)
                x = torch.clamp(x, -128, 127)

                rexp=-6*torch.ones_like(rexp)
            # running_exp=tmp = torch.round(torch.log2(self.quant.desired_delta))
            return x

        
    def get_weight_factor(self):
        return self.alpha[:, None, None, None]
