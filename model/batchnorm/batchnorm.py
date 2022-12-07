from typing import Union

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from ..Quantizer import LinQuantExpScale

from ..logger import logger_init, logger_forward

from ..DataWrapper import DataWrapper

from .. import (
    __DEBUG__,
    __HIGH_PRES__,
    __HIGH_PRES_USE_RUNNING__,
)

from .. import __TESTING_FLAGS__


from .functions import calculate_alpha, calculate_alpha_fixed, calculate_n, calculate_n_fixed, calculate_t

NAME_INDEX = 0


class BatchNorm2d(torch.nn.BatchNorm2d):
    """
    BatchNorm2d Modified nn.BatchNorm2d

    Modified to create a convolution weight adaptation factor and calculate the eval BN as a addition and shift

    :param num_features: Number of channels
    :type num_features: int
    :param eps: A factor to make div 0 impossible, defaults to 0.00001
    :type eps: float, optional
    :param momentum: The momentum of th BN, defaults to 0.1
    :type momentum: float, optional
    :param affine: BN Affine, defaults to True
    :type affine: bool, optional
    :param track_running_stats: BN running stats, defaults to True
    :type track_running_stats: bool, optional

    :param fixed_n: Set the shift to a layer-wise value rather than channel-wise, defaults to False
    :type fixed_n: bool, optional

    :param out_quant:  A callable object which overrides the default output quantization, gets called with (values) , defaults to None
    :type out_quant: _type_, optional
    :param out_quant_bits: Number of bits for the output quantization, defaults to 8
    :type out_quant_bits: int, optional
    :param out_quant_channel_wise: Channel-wise output quantization, defaults to False
    :type out_quant_channel_wise: bool, optional
    :param out_quant_args:  Overrides arguments for the out quantization initializer with custom ones, defaults to None
    :type out_quant_args: _type_, optional
    :param out_quant_kargs: Passes named arguments to the initializer of the out quantization class, defaults to {}
    :type out_quant_kargs: dict, optional
    """

    @logger_init
    def __init__(
        self,
        num_features: int,
        eps: float = 0.00001,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
        fixed_n: bool = False,
        out_quant=None,
        out_quant_args=None,
        out_quant_kargs={},
    ):
        """
        Please read the class help
        """

        super(BatchNorm2d, self).__init__(num_features, eps, momentum, affine, track_running_stats, device, dtype)

        self.register_buffer("n", torch.zeros(1, num_features, 1, 1))
        self.register_buffer("t", torch.zeros(1, num_features, 1, 1))
        self.register_buffer("alpha", 1.0 / np.sqrt(2.0) * torch.ones(num_features))

        self.func_t = calculate_t
        self.fixed_n = fixed_n
        if fixed_n:
            self.func_n = calculate_n_fixed
            self.func_a = calculate_alpha_fixed
        else:
            self.func_n = calculate_n
            self.func_a = calculate_alpha

        if out_quant_args == None:
            out_quant_args = (
                8,
                (1, num_features, 1, 1),
            )

        if out_quant == None:
            self.out_quant = LinQuantExpScale(*out_quant_args, **out_quant_kargs)
        else:
            self.out_quant = out_quant(*out_quant_args, **out_quant_kargs)

        # TODO: Delete
        global NAME_INDEX
        self.NAME_INDEX = NAME_INDEX
        NAME_INDEX += 1
        self.FILE_NAME = './bn_values/' + str(self.NAME_INDEX)
        self.counter_max = 1000
        self.counter = self.counter_max
        self.STAFFEL = self.NAME_INDEX * 300
        self.register_buffer('mul_norm', torch.ones_like(self.running_var))

    def get_weight_factor(self):
        """
        get_weight_factor Returns a function to calculate alpha with a singe value
        """

        def ret_fun(rexp):
            self.alpha = self.func_a(
                weight=self.weight.view(-1).detach(),
                mean=self.running_mean.view(-1).detach(),
                var=self.running_var.view(-1).detach(),
                out_quant=self.out_quant.delta_out.view(-1).detach(),
                rexp=rexp.view(-1).detach(),
            )
            return self.alpha[:, None, None, None]

        return ret_fun

    @logger_forward
    def pre_forward(self, conv=None) -> None:
        if __TESTING_FLAGS__['FUZE_BN']:
            if self.training:
                # if self.counter == self.counter_max:
                if self.counter % 100 == 0:
                    if conv is not None:
                        print('fuzing bn')
                        mod_fac = self.weight.data.view(-1).div(self.running_var.data.add(self.eps).sqrt().view(-1))
                        mod_fac = mod_fac.view(-1, 1, 1, 1)
                        self.mul_norm *= self.weight.data.square()
                        self.running_var.data = torch.ones_like(self.running_var)
                        conv.weight.data = conv.weight.data * mod_fac
                        self.weight.data = torch.ones_like(self.weight)
                self.counter -= 1

    @logger_forward
    def forward(self, input: DataWrapper, activation: Union[None, nn.Module] = None, conv=None) -> DataWrapper:

        if not self.training:
            return self.forward_eval(input, activation)
        return self.forward_train_fast(input, activation)
        return self.forward_train_test(input, activation)

        # x, rexp = input.get()

        # if activation != None:
        #     self.out_quant.copy(activation)
        #     quant = activation
        # else:
        #     quant = self.out_quant

        # # TODO:
        # #   define something to freeze the bn

        # if self.training:
        #     xorig = x.data.clone()
        #     if __HIGH_PRES__:
        #         xorig = x.data.clone().detach()

        #     # # TODO:Delete
        #     # var = torch.var(x.detach(), [0, 2, 3], unbiased=False, keepdim=True)
        #     # mu = torch.mean(x.detach(), [0, 2, 3], keepdim=True)
        #     # with open(self.FILE_NAME + '_var.csv', 'a+') as f:
        #     #     np.savetxt(f, var.detach().cpu().numpy().reshape(1, -1))
        #     # with open(self.FILE_NAME + '_mu.csv', 'a+') as f:
        #     #     np.savetxt(f, mu.detach().cpu().numpy().reshape(1, -1))

        #     # with open(self.FILE_NAME + '_bias.csv', 'a+') as f:
        #     #     np.savetxt(f, self.bias.detach().cpu().numpy().reshape(1, -1))
        #     # with open(self.FILE_NAME + '_weight.csv', 'a+') as f:
        #     #     np.savetxt(f, self.weight.detach().cpu().numpy().reshape(1, -1))

        #     # until here
        #     if not __TESTING_FLAGS__['FREEZE_BN']:  # or self.STAFFEL > 0:
        #         x = super().forward(x)
        #         # var = torch.var(x, [0, 2, 3], unbiased=False, keepdim=True).div(self.mul_norm.view(1,-1,1,1))
        #         # mu = torch.mean(x, [0, 2, 3], keepdim=True)
        #         # mul = self.weight.view(1, -1, 1, 1).div(var.add(self.eps).sqrt().view(1, -1, 1, 1))
        #         # with torch.no_grad():
        #         #      self.running_var =  self.running_var*(1-self.momentum)+var.view(-1)*self.momentum
        #         #      self.running_mean =  self.running_mean*(1-self.momentum)+mu.view(-1)*self.momentum
        #         # # x = super(BatchNorm2d, self).forward(x)
        #         # x = x.sub(mu.view(1, -1, 1, 1)).mul(mul).add(self.bias.view(1, -1, 1, 1))

        #     else:
        #         if self.counter == 0:
        #             print('bn frozen')
        #         # self.weight.requires_grad_(False)
        #         var = torch.var(x, [0, 2, 3], unbiased=False,)
        #         mu = torch.mean(x, [0, 2, 3],)
        #         mom = 1  # if self.counter < 0 else (self.counter / self.counter_max)*(1-0.01)+0.01
        #         # mu.data = self.running_mean.data * (1 - mom) + mu * mom
        #         var = self.running_var.data * (1 - mom) + var.div(self.mul_norm.view(-1)) * mom
        #         # running_mean_tmp = mu
        #         # running_var_tmp = var
        #         mul = self.weight.view(1, -1, 1, 1).div(var.add(self.eps).sqrt().view(1, -1, 1, 1))

        #         x = x.sub(mu.view(1, -1, 1, 1)).mul(mul).add(self.bias.view(1, -1, 1, 1))
        #         self.running_mean.data = mu
        #         self.running_var.data = var
        #         self.counter -= 1

        #     # # # TODO:Delete
        #     # with open(self.FILE_NAME + '_running_var.csv', 'a+') as f:
        #     #     np.savetxt(f, self.running_var.detach().cpu().numpy().reshape(1, -1))
        #     # with open(self.FILE_NAME + '_running_mu.csv', 'a+') as f:
        #     #     np.savetxt(f, self.running_mean.detach().cpu().numpy().reshape(1, -1))
        #     # # # until here

        #     if not __HIGH_PRES__:  # and False:
        #         x = quant(x, False, input)
        #     else:
        #         x = quant(x, True, input)
        #         with torch.no_grad():
        #             if __HIGH_PRES_USE_RUNNING__:
        #                 mu = self.running_mean.clone()
        #                 var = self.running_var.clone()
        #             else:
        #                 var = torch.var(xorig, [0, 2, 3], unbiased=False, keepdim=True)
        #                 mu = torch.mean(xorig, [0, 2, 3], keepdim=True)

        #             n = self.weight.view(-1) / (quant.delta_in.view(-1) * torch.sqrt(var.view(-1) + self.eps))
        #             n = n.view(1, -1, 1, 1)
        #             nr = self.func_n(
        #                 weight=torch.abs(self.weight.view(-1)),
        #                 bias=self.bias.view(-1),
        #                 mean=self.running_mean.view(-1),
        #                 var=self.running_var.view(-1),
        #                 out_quant=quant.delta_in.view(-1),
        #                 rexp=rexp.view(-1),
        #             ).detach()

        #             t = self.func_t(
        #                 weight=self.weight.view(-1),
        #                 bias=self.bias.view(-1),
        #                 mean=self.running_mean.view(-1),
        #                 var=self.running_var.view(-1),
        #                 out_quant=quant.delta_in.view(-1),
        #                 rexp=rexp.view(-1),
        #                 n=self.n.view(-1),
        #             ).detach()

        #             tmp = torch.exp2(nr.view(1, -1, 1, 1))

        #             # t = t.view(1, -1, 1, 1).floor().clamp(-128,127)

        #             # xorig = (
        #             #     xorig.mul_(n.view(1, -1, 1, 1))
        #             #     .floor_()
        #             #     .clamp(-128,127)
        #             #     .add_(t.view(1, -1, 1, 1))
        #             #     .clamp_(min=quant.min, max=quant.max)
        #             #     .mul_(quant.delta_out)
        #             # )
        #             t = t.view(1, -1, 1, 1).div(tmp).floor().mul(tmp)

        #             xorig = (
        #                 xorig.mul_(n.view(1, -1, 1, 1))
        #                 .add_(t.view(1, -1, 1, 1))
        #                 .floor_()
        #                 .clamp_(min=quant.min, max=quant.max)
        #                 .mul_(quant.delta_out)
        #             )
        #         x.data = xorig

        #     # var = torch.var(x.detach(), [0, 2, 3], unbiased=False, keepdim=True)
        #     # mu = torch.mean(x.detach(), [0, 2, 3], keepdim=True)
        #     # with open(self.FILE_NAME + '_out_var.csv', 'a+') as f:
        #     #     np.savetxt(f, var.detach().cpu().numpy().reshape(1, -1))
        #     # with open(self.FILE_NAME + '_out_mu.csv', 'a+') as f:
        #     #     np.savetxt(f, mu.detach().cpu().numpy().reshape(1, -1))

        #     rexp = torch.log2(quant.delta_out)
        #     return input.set(x, rexp)

        # else:
        #     with torch.no_grad():
        #         self.n = self.func_n(
        #             weight=torch.abs(self.weight.view(-1)),
        #             bias=self.bias.view(-1),
        #             mean=self.running_mean.view(-1),
        #             var=self.running_var.view(-1),
        #             out_quant=quant.delta_in.view(-1),
        #             rexp=rexp.view(-1),
        #         ).detach().view(1, -1, 1, 1)

        #         t = self.func_t(
        #             weight=self.weight.view(-1),
        #             bias=self.bias.view(-1),
        #             mean=self.running_mean.view(-1),
        #             var=self.running_var.view(-1),
        #             out_quant=quant.delta_in.view(-1),
        #             rexp=rexp.view(-1),
        #             n=self.n.view(-1),
        #         ).detach()

        #         tmp = torch.exp2(self.n.view(1, -1, 1, 1))

        #         self.t = t.view(1, -1, 1, 1).div(tmp).floor()
        #         x = x + self.t
        #         x = x.mul(tmp.view(1, -1, 1, 1))

        #         # self.t = t.view(1, -1, 1, 1).floor().clamp(-128,127)
        #         # x = x.mul(tmp.view(1, -1, 1, 1)).floor().clamp(-128,127)
        #         # x = x + self.t

        #         x = x.floor()
        #         x = x.clamp(quant.min, quant.max)
        #         if __DEBUG__:
        #             x = torch.nan_to_num(x)

        #         rexp = torch.log2(quant.delta_out)
        #         return input.set(x, rexp)

    @logger_forward
    def forward_train_fast(self, input: DataWrapper, activation: Union[None, nn.Module] = None):
        x, rexp = input.get()

        if activation != None:
            self.out_quant.copy(activation)
            quant = activation
        else:
            quant = self.out_quant

        x = super(BatchNorm2d, self).forward(x)
        x = quant(x, False, input)

        rexp = torch.log2(quant.delta_out)
        return input.set(x, rexp)

    @logger_forward
    def forward_train_test(self, input: DataWrapper, activation: Union[None, nn.Module] = None):
        x, rexp = input.get()

        if activation != None:
            self.out_quant.copy(activation)
            quant = activation
        else:
            quant = self.out_quant

        ###### SAVE VALUES #####
        var_u = torch.var(x.detach(), [0, 2, 3], unbiased=False, keepdim=True)
        mu = torch.mean(x.detach(), [0, 2, 3], keepdim=True)
        with open(self.FILE_NAME + '_var.csv', 'a+') as f:
            np.savetxt(f, var_u.detach().cpu().numpy().reshape(1, -1))
        with open(self.FILE_NAME + '_mu.csv', 'a+') as f:
            np.savetxt(f, mu.detach().cpu().numpy().reshape(1, -1))

        with open(self.FILE_NAME + '_bias.csv', 'a+') as f:
            np.savetxt(f, self.bias.detach().cpu().numpy().reshape(1, -1))
        with open(self.FILE_NAME + '_weight.csv', 'a+') as f:
            np.savetxt(f, self.weight.detach().cpu().numpy().reshape(1, -1))

        ###### CALCULATE #######
        if not __TESTING_FLAGS__['FREEZE_BN']:  # or self.STAFFEL > 0:
            var_u = torch.var(x, [0, 2, 3], unbiased=False, keepdim=True).div(self.mul_norm.view(1, -1, 1, 1))
            mu = torch.mean(x, [0, 2, 3], keepdim=True)
            mul = self.weight.view(1, -1, 1, 1).div(var_u.add(self.eps).sqrt().view(1, -1, 1, 1))
            with torch.no_grad():
                self.running_var = self.running_var * (1 - self.momentum) + var_u.view(-1) * self.momentum
                self.running_mean = self.running_mean * (1 - self.momentum) + mu.view(-1) * self.momentum
            # x = super(BatchNorm2d, self).forward(x)

            with torch.no_grad():
                N = list(x.shape)
                del N[1]
                N = np.prod(N)
                # val = (self.weight.view(1,-1,1,1)/(torch.sqrt(var.view(1,-1,1,1)+self.eps))).view(1,-1,1,1)
                # print(val.shape)
                val = (1 - 1 / N - 1 / (N - 1) * ((x - mu.view(1, -1, 1, 1)) /
                       torch.sqrt(var_u + self.eps).view(1, -1, 1, 1)).square())
                val = val.detach().cpu().numpy().reshape(-1)
                arr = [np.min(val), np.mean(val), np.max(val)]
                arr = np.array(arr)
                with open(self.FILE_NAME + '_backprop.csv', 'a+') as f:
                    np.savetxt(f, arr.reshape(1, -1))

            x = x.sub(mu.view(1, -1, 1, 1)).mul(mul).add(self.bias.view(1, -1, 1, 1))

        else:
            if self.counter == 0:
                print('bn frozen')
            # self.weight.requires_grad_(False)
            var = torch.var(x, [0, 2, 3], unbiased=False,)
            mu = torch.mean(x, [0, 2, 3],)

            mom = 0.01 if self.counter < 0 else (self.counter / self.counter_max) * (1 - 0.01) + 0.01
            # mu.data = self.running_mean.data * (1 - mom) + mu * mom
            # var_u = var.clone()
            var_u = self.running_var.data * (1 - mom) + var.div(self.mul_norm.view(-1)) * mom
            # mu_u = self.running_mean.data * (1 - mom) + mu.div(self.mul_norm.view(-1)) * mom
            # running_mean_tmp = mu
            # running_var_tmp = var
            mul = self.weight.view(1, -1, 1, 1).div(var_u.add(self.eps).sqrt().view(1, -1, 1, 1))

            with torch.no_grad():
                N = list(x.shape)
                del N[1]
                N = np.prod(N)
                # val = (self.weight.view(1,-1,1,1)/(torch.sqrt(var.view(1,-1,1,1)+self.eps))).view(1,-1,1,1)
                # print(val.shape)
                val = (1 - 1 / N - 1 / (N - 1) * ((x - mu.view(1, -1, 1, 1)) /
                       torch.sqrt(var_u + self.eps).view(1, -1, 1, 1)).square())
                val = val.detach().cpu().numpy().reshape(-1)
                arr = [np.min(val), np.mean(val), np.max(val)]
                arr = np.array(arr)
                with open(self.FILE_NAME + '_backprop.csv', 'a+') as f:
                    np.savetxt(f, arr.reshape(1, -1))

            x = x.sub(mu.view(1, -1, 1, 1)).mul(mul).add(self.bias.view(1, -1, 1, 1))
            if mom <= 0.1:
                # self.running_mean.data = mu
                self.running_mean.data = self.running_mean.data * (1 - 0.1) + mu * 0.1
                self.running_var.data = var_u
            else:
                self.running_mean.data = self.running_mean.data * (1 - 0.1) + mu * 0.1
                self.running_var.data = self.running_var.data * (1 - 0.1) + var.div(self.mul_norm.view(-1)) * 0.1
            self.counter -= 1

        ###### SAVE VALUES #####
        with open(self.FILE_NAME + '_running_var.csv', 'a+') as f:
            np.savetxt(f, self.running_var.detach().cpu().numpy().reshape(1, -1))
        with open(self.FILE_NAME + '_running_mu.csv', 'a+') as f:
            np.savetxt(f, self.running_mean.detach().cpu().numpy().reshape(1, -1))

        x = quant(x, False, input)

        rexp = torch.log2(quant.delta_out)
        return input.set(x, rexp)

    @logger_forward
    def forward_eval(self, input: DataWrapper, activation: Union[None, nn.Module] = None):
        x, rexp = input.get()

        if activation != None:
            self.out_quant.copy(activation)
            quant = activation
        else:
            quant = self.out_quant

        with torch.no_grad():
            self.n = self.func_n(
                weight=torch.abs(self.weight.view(-1)),
                bias=self.bias.view(-1),
                mean=self.running_mean.view(-1),
                var=self.running_var.view(-1),
                out_quant=quant.delta_in.view(-1),
                rexp=rexp.view(-1),
            ).detach().view(1, -1, 1, 1)

            t = self.func_t(
                weight=self.weight.view(-1),
                bias=self.bias.view(-1),
                mean=self.running_mean.view(-1),
                var=self.running_var.view(-1),
                out_quant=quant.delta_in.view(-1),
                rexp=rexp.view(-1),
                n=self.n.view(-1),
            ).detach()

            tmp = torch.exp2(self.n.view(1, -1, 1, 1))

            self.t = t.view(1, -1, 1, 1).div(tmp).floor()
            x = x + self.t
            x = x.mul(tmp.view(1, -1, 1, 1))

            x = x.floor()
            x = x.clamp(quant.min, quant.max)
            if __DEBUG__:
                x = torch.nan_to_num(x)

            rexp = torch.log2(quant.delta_out)
            return input.set(x, rexp)
