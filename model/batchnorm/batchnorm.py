from typing import Union
import torch
import torch.nn as nn
import numpy as np

from ..Quantizer import LinQuantExpScale

from ..logger import logger_init, logger_forward

from ..Type import Data_wrapper

from .. import (
    __DEBUG__,
    __HIGH_PRES__,
    __HIGH_PRES_USE_RUNNING__,
)


from .functions import calculate_alpha, calculate_alpha_fixed, calculate_n, calculate_n_fixed, calculate_t


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
        out_quant_bits: int = 8,
        out_quant_channel_wise: bool = False,
        out_quant_args=None,
        out_quant_kargs={},
    ):

        """
        Please read the class help
        """

        super(BatchNorm2d, self).__init__(num_features, eps, momentum, affine, track_running_stats, device, dtype)

        self.register_buffer("n", torch.zeros(num_features))
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
                out_quant_bits,
                (-1,) if not out_quant_channel_wise else (1, num_features, 1, 1),
                0.1,
                "floor",
            )

        if out_quant == None:
            self.out_quant = LinQuantExpScale(*out_quant_args, **out_quant_kargs)
        else:
            self.out_quant = out_quant

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
    def forward(self, input: Data_wrapper, activation: Union[None, nn.Module] = None) -> Data_wrapper:
        x, rexp = input.get()

        if activation != None:
            self.out_quant.copy(activation)
            quant = activation
        else:
            quant = self.out_quant

        # TODO:
        #   define something to freeze the bn

        if self.training:
            if __HIGH_PRES__:
                xorig = x.data.clone().detach()

            x = super(BatchNorm2d,self).forward(x)

            if not __HIGH_PRES__:
                x = quant(x, False)
            else:
                x = quant(x, True)
                with torch.no_grad():
                    if __HIGH_PRES_USE_RUNNING__:
                        mu = self.running_mean.clone()
                        var = self.running_var.clone()
                    else:
                        var = torch.var(xorig, [0, 2, 3], unbiased=False, keepdim=True)
                        mu = torch.mean(xorig, [0, 2, 3], keepdim=True)

                    n = self.weight.view(-1) / (quant.delta_in.view(-1) * torch.sqrt(var.view(-1) + self.eps))
                    nr = self.func_n(
                        weight=torch.abs(self.weight.view(-1)),
                        bias=self.bias.view(-1),
                        mean=self.running_mean.view(-1),
                        var=self.running_var.view(-1),
                        out_quant=quant.delta_in.view(-1),
                        rexp=rexp.view(-1),
                    ).detach()

                    t = self.func_t(
                        weight=self.weight.view(-1),
                        bias=self.bias.view(-1),
                        mean=self.running_mean.view(-1),
                        var=self.running_var.view(-1),
                        out_quant=quant.delta_in.view(-1),
                        rexp=rexp.view(-1),
                        n=self.n.view(-1),
                    ).detach()

                    tmp = torch.exp2(nr.view(1, -1, 1, 1))

                    t = t.view(1, -1, 1, 1).div(tmp).floor().mul(tmp)

                    xorig = (
                        xorig.mul_(n.view(1, -1, 1, 1))
                        .add_(t.view(1, -1, 1, 1))
                        .floor_()
                        .clamp_(min=quant.min, max=quant.max)
                        .mul_(quant.delta_out)
                    )
                x.data = xorig

            rexp = torch.log2(quant.delta_out)
            return input.set(x, rexp)

        else:
            with torch.no_grad():
                self.n = self.func_n(
                    weight=torch.abs(self.weight.view(-1)),
                    bias=self.bias.view(-1),
                    mean=self.running_mean.view(-1),
                    var=self.running_var.view(-1),
                    out_quant=quant.delta_in.view(-1),
                    rexp=rexp.view(-1),
                ).detach()

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
