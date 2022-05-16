import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model_old.quantizer import *
from model_old.utils import*

get_weight_factor_counter=0

class BatchNorm2dQuant(nn.Module):  # TODO QUANTIZE
    def __init__(self, num_features, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(BatchNorm2dQuant, self).__init__()

        self.weight = torch.nn.Parameter(
            torch.empty(num_features, **factory_kwargs))
        self.bias = torch.nn.Parameter(
            torch.empty(num_features, **factory_kwargs))
        torch.nn.init.ones_(self.weight)
        torch.nn.init.zeros_(self.bias)
        self.register_buffer('n', torch.zeros(num_features))
        self.register_buffer('t', torch.zeros(num_features))

        # req from last cycle
        self.register_buffer('sig', torch.ones(num_features))
        self.sig_for_alpha = 1
        self.register_buffer('mu', torch.zeros(num_features))

        self.first = True

        # from quant
        self.quant = LinQuant(8)
        self.register_buffer('inference_n', torch.ones(num_features))

        # for weights
        self.register_buffer('alpha', torch.ones(num_features))

        # for debugging
        self.old_sigs = []
        self.true_old_suigs = []
        self.old_alpha = []

    def plot(self):
        import numpy as np
        tmp=[]
        tmp2=[]
        tmp3 = []
        for i in range(len(self.old_sigs)):
            tmp.append(self.old_sigs[i].cpu().numpy())
        tmp = np.array(tmp)

        for i in range(len(self.old_sigs)):
            tmp2.append(self.true_old_suigs[i].cpu().numpy())
        tmp2 = np.array(tmp2)

        for i in range(len(self.old_alpha)):
            tmp3.append(self.old_alpha[i].cpu().numpy())
        tmp3 = np.array(tmp3)

        import matplotlib.pyplot as plt
        plt.plot(tmp)
        plt.plot(tmp2, 'o')
        plt.plot(tmp3,'--')
        plt.show()

    def forward(self, x, in_quant=1):
        if self.training:
            if self.training:
                mu = x.mean([0, 2, 3])
                sig = x.var([0, 2, 3], unbiased=False)
                with torch.no_grad():
                    self.mu = self.mu*0.9 + 0.1*mu.squeeze()
                    self.sig = self.sig*0.9 + 0.1*sig.squeeze()
                    self.old_sigs.append(self.sig.cpu())
                    self.true_old_suigs.append(sig.cpu())
                    self.old_alpha.append(self.alpha.cpu())
                    if torch.any(torch.isnan(self.sig)):
                        self.sig = sig
                    if torch.any(torch.isnan(self.mu)):
                        self.mu = mu
                    if self.first:
                        self.first = False
                        self.sig = sig
                        self.mu = mu
            else:
                mu = self.mu
                sig = self.sig

            xorig = x.clone()
            # clamp to min 0 so n can't be negative
            weights_used = self.weight.clamp(0)

            x = (x-mu[None, :, None, None]) / \
                (torch.sqrt(sig[None, :, None, None]+1e-5))
            x = x*weights_used[None, :, None, None] + \
                self.bias[None, :, None, None]

            x = self.quant(x)
            x = (x)/self.quant.delta

            with torch.no_grad():
                n = (weights_used*in_quant.view(-1))/(torch.sqrt(self.sig+1e-5)* self.quant.delta)
                # n = (weights_used)/(torch.sqrt(sig+1e-5)* self.quant.delta)
                self.n =torch.round(torch.log2(n))  # torch.round(n)
                # print(torch.mean(self.n))
                # + 1.0/self.quant.desired_delta
                self.t = -mu*(weights_used)/(torch.sqrt(self.sig+1e-5) * self.quant.delta) + self.bias/self.quant.delta
                self.t = torch.round(self.t).clamp(-128, 127)
                self.sig_for_alpha = sig

            xorig = xorig * torch.exp2(self.n)[None, :, None, None]/in_quant.view(-1)[None, :, None, None] + self.t[None, :, None, None]
            # xorig = xorig * torch.exp2(self.n)[None, :, None, None] + self.t[None, :, None, None]
            xorig = torch.round(xorig)
            xorig = torch.clamp(xorig, -128, 127)

            # print('diff',torch.max(torch.abs(x-xorig)))
            # print('xorig',torch.max(torch.abs(xorig)))
            # print('x',torch.max(torch.abs(xorig)))
            x, xorig = switch.apply(x, xorig)
            tmp = torch.round(torch.log2(self.quant.delta))

            set_rexp(-6)

            x = x*(2**get_rexp())

            # x = x/2**6
            # x = x*torch.exp2(tmp)
            return x
        else:
            mu = self.mu
            sig = self.sig
            # clamp to min 0 so n can't be negative
            weights_used = self.weight.clamp(0)
            n = (weights_used*in_quant.view(-1))/(torch.sqrt(sig+1e-5) * self.quant.delta)
            self.n = torch.round(torch.log2(n))  # torch.round(n)
            # print(torch.mean(self.n))
            # + 1.0/self.quant.desired_delta
            self.t = -mu*(weights_used)/(torch.sqrt(sig+1e-5) * self.quant.delta) + self.bias/self.quant.delta
            self.t = torch.round(self.t).clamp(-128, 127)

            tmp_n = self.n+get_rexp()

            self.inference_n = tmp_n

            x = x*torch.exp2(tmp_n)[None, :, None, None] + self.t[None, :, None, None]
            x = torch.round(x)
            x = torch.clamp(x, -128, 127)

            set_rexp(-6)
            # running_exp=tmp = torch.round(torch.log2(self.quant.desired_delta))
            return x

        
    def get_weight_factor(self,in_delta):
        mom = 0
        ones = torch.ones_like(self.alpha)

        global get_weight_factor_counter
        if self.training:
            with torch.no_grad():
                sig = (self.weight*in_delta.view(-1)/self.quant.delta).square() * torch.exp2(-2*self.n)-1e-5
                alpha = torch.sqrt(sig/self.sig)
                # print(alpha)
                alpha = alpha.masked_fill(torch.isnan(alpha), 1)
                old_alpha = self.alpha
                self.alpha = mom*self.alpha + (1-mom)*alpha*self.alpha
                # self.alpha = self.alpha.clamp(0.5,2)
                bounding_fact = np.sqrt(2)*1.1
                cond1 = self.alpha < bounding_fact
                cond2 = self.alpha > 1/bounding_fact
                # cond = torch.logical_or(cond1,cond2)
                # self.alpha = torch.where(cond,self.alpha,ones)
                self.alpha = self.alpha.clamp(0.125, 8)
                self.alpha = torch.where(cond1, self.alpha, self.alpha/2)
                self.alpha = torch.where(cond2, self.alpha, self.alpha*2)

                # update sig
                self.sig = self.sig*torch.square(self.alpha/old_alpha)

                # if torch.any(~cond1) or torch.any(~cond2):
                #     print("Flip happend")
                get_weight_factor_counter+=torch.sum(~cond1)+torch.sum(~cond2)
        else:
            if get_weight_factor_counter!=0:
                print(f"\n\n{get_weight_factor_counter} flips happend\n")
                get_weight_factor_counter=0
        return self.alpha[:, None, None, None]
        # return ones[:, None,None, None]
