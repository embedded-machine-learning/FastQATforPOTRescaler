# Generic Type imports
from operator import xor
from types import FunctionType
from typing import Tuple

# Torch imports
import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t, Tensor

# numpy
import numpy as np

# current module imports
from .quantizer import Quant, get_abs, FakeQuant, LinQuantExpScale

# Global information imports
from . import (
    __DEBUG__,
    LOG,
    __LOG_LEVEL_IMPORTANT__,
    __LOG_LEVEL_NORMAL__,
    __LOG_LEVEL_DEBUG__,
    __LOG_LEVEL_HIGH_DETAIL__,
    __LOG_LEVEL_TO_MUCH__,
    __HIGH_PRES__,
    __HIGH_PRES_USE_RUNNING__,
)


#############################################################################
#                       FUNCTION                                            #
#############################################################################


def calculate_n(
    weight: Tensor,
    bias: Tensor,
    mean: Tensor,
    var: Tensor,
    out_quant: Tensor,
    rexp: Tensor,
) -> Tensor:
    """
    calculate_n calculates the shift factor

    :param weight: The absolute of the weight vector
    :type weight: Tensor
    :param bias: The bias of the BN
    :type bias: Tensor
    :param mean: The mean of the BN
    :type mean: Tensor
    :param var: The variance of the BN
    :type var: Tensor
    :param out_quant: The out quantization factor
    :type out_quant: Tensor
    :param rexp: The input exponent
    :type rexp: Tensor
    :return: Return the shift factor
    :rtype: Tensor
    """
    with torch.no_grad():
        n = torch.log2(weight.abs() / (out_quant * torch.sqrt(var + 1e-5))) + rexp.view(-1)
        LOG(__LOG_LEVEL_HIGH_DETAIL__, "calculate_n: n", n)
        n = n.ceil()
        LOG(__LOG_LEVEL_HIGH_DETAIL__, "calculate_n: n post ceil", n)
        return n


def calculate_n_fixed(
    weight: Tensor,
    bias: Tensor,
    mean: Tensor,
    var: Tensor,
    out_quant: Tensor,
    rexp: Tensor,
) -> Tensor:
    """
    calculate_n calculates the shift factor for a whole layer

    :param weight: The absolute of the weight vector
    :type weight: Tensor
    :param bias: The bias of the BN
    :type bias: Tensor
    :param mean: The mean of the BN
    :type mean: Tensor
    :param var: The variance of the BN
    :type var: Tensor
    :param out_quant: The out quantization factor
    :type out_quant: Tensor
    :param rexp: The input exponent
    :type rexp: Tensor
    :return: Return the shift factor
    :rtype: Tensor
    """
    with torch.no_grad():
        n = torch.log2(weight.abs() / (out_quant * torch.sqrt(var + 1e-5))) + rexp.view(-1)
        LOG(__LOG_LEVEL_HIGH_DETAIL__, "calculate_n_fixed: n", n)
        nr = n.max() * torch.ones_like(n)
        LOG(__LOG_LEVEL_HIGH_DETAIL__, "calculate_n_fixed: nr", nr)
        nr = nr.ceil()
        LOG(__LOG_LEVEL_HIGH_DETAIL__, "calculate_n_fixed: nr post ceil", nr)
        return nr


def calculate_t(
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    var: torch.Tensor,
    out_quant: torch.Tensor,
    rexp: torch.Tensor,
    n: torch.Tensor,
) -> torch.Tensor:
    """
    calculate_t calculates the additive value

    :param weight: The absolute of the weight vector
    :type weight: Tensor
    :param bias: The bias of the BN
    :type bias: Tensor
    :param mean: The mean of the BN
    :type mean: Tensor
    :param var: The variance of the BN
    :type var: Tensor
    :param out_quant: The out quantization factor
    :type out_quant: Tensor
    :param rexp: The input exponent
    :type rexp: Tensor
    :return: Return the additive value
    :rtype: Tensor
    """
    with torch.no_grad():
        t = -mean * (weight / (out_quant * torch.sqrt(var + 1e-5))) + bias / out_quant
        LOG(__LOG_LEVEL_HIGH_DETAIL__, "calculate_t: t", t)
        return t


def calculate_alpha(
    weight: torch.Tensor,
    mean: torch.Tensor,
    var: torch.Tensor,
    out_quant: torch.Tensor,
    rexp: torch.Tensor,
) -> torch.Tensor:
    """
    calculate_alpha calculates the adaptation factor for the weights

    :param weight: The absolute of the weight vector
    :type weight: Tensor
    :param mean: The mean of the BN
    :type mean: Tensor
    :param var: The variance of the BN
    :type var: Tensor
    :param out_quant: The out quantization factor
    :type out_quant: Tensor
    :param rexp: The input exponent
    :type rexp: Tensor
    :return: The adaptation factor
    :rtype: torch.Tensor
    """
    with torch.no_grad():
        n = weight.abs() * rexp.view(-1)
        n = n.div_(out_quant).div_(torch.sqrt(var + 1e-5))
        n = n.log2_()
        # n = torch.log2(weight.abs() * rexp.view(-1) / (out_quant * torch.sqrt(var + 1e-5)))
        LOG(__LOG_LEVEL_HIGH_DETAIL__, "calculate_alpha: n", n)
        nr = torch.ceil(n)
        LOG(__LOG_LEVEL_HIGH_DETAIL__, "calculate_alpha: nr", nr)
        alpha = torch.sign(weight) * torch.exp2(n - nr)
        LOG(__LOG_LEVEL_HIGH_DETAIL__, "calculate_alpha: alpha", alpha)
    return alpha


def calculate_alpha_fixed(
    weight: torch.Tensor,
    mean: torch.Tensor,
    var: torch.Tensor,
    out_quant: torch.Tensor,
    rexp: torch.Tensor,
) -> torch.Tensor:
    """
    calculate_alpha calculates the adaptation factor for the weights for a fixed n

    :param weight: The absolute of the weight vector
    :type weight: Tensor
    :param mean: The mean of the BN
    :type mean: Tensor
    :param var: The variance of the BN
    :type var: Tensor
    :param out_quant: The out quantization factor
    :type out_quant: Tensor
    :param rexp: The input exponent
    :type rexp: Tensor
    :return: The adaptation factor
    :rtype: torch.Tensor
    """
    with torch.no_grad():
        n = torch.log2(weight.abs() / (out_quant * torch.sqrt(var + 1e-5))) + rexp.view(-1)
        LOG(__LOG_LEVEL_HIGH_DETAIL__, "calculate_alpha_fixed: n", n)
        nr = n.max() * torch.ones_like(n)
        LOG(__LOG_LEVEL_HIGH_DETAIL__, "calculate_alpha_fixed: nr", nr)
        nr = torch.ceil(nr)
        LOG(__LOG_LEVEL_HIGH_DETAIL__, "calculate_alpha_fixed: nr post ceil", nr)
        alpha = torch.sign(weight) * torch.exp2(n - nr)
        LOG(__LOG_LEVEL_HIGH_DETAIL__, "calculate_alpha_fixed: alpha", alpha)
    raise NotImplementedError()
    return alpha


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
        LOG(
            __LOG_LEVEL_DEBUG__,
            f"BatchNorm2d passed arguments:\n\
            num_features:                   {num_features}\n\
            eps:                            {eps}\n\
            momentum:                       {momentum}\n\
            affine:                         {affine}\n\
            track_running_stats:            {track_running_stats}\n\
            device:                         {device}\n\
            dtype:                          {dtype}\n\
            fixed_n:                        {fixed_n}\n\
            out_quant:                      {out_quant}\n\
            out_quant_bits:                 {out_quant_bits}\n\
            out_quant_channel_wise:         {out_quant_channel_wise}\n\
            out_quant_args:                 {out_quant_args}\n\
            out_quant_kargs:                {out_quant_kargs}\n\
            ",
        )

        super(BatchNorm2d, self).__init__(num_features, eps, momentum, affine, track_running_stats, device, dtype)

        self.register_buffer("n", torch.zeros(num_features))
        LOG(__LOG_LEVEL_TO_MUCH__, "BatchNorm2d.__init__: buffer n", self.n)
        self.register_buffer("t", torch.zeros(1,num_features,1,1))
        LOG(__LOG_LEVEL_TO_MUCH__, "BatchNorm2d.__init__: buffer t", self.t)
        self.register_buffer("alpha", 1.0 / np.sqrt(2.0) * torch.ones(num_features))
        LOG(__LOG_LEVEL_TO_MUCH__, "BatchNorm2d.__init__: buffer alpha", self.alpha)

        self.func_t = calculate_t
        self.fixed_n = fixed_n
        if fixed_n:
            self.func_n = calculate_n_fixed
            self.func_a = calculate_alpha_fixed
        else:
            self.func_n = calculate_n
            self.func_a = calculate_alpha
        LOG(__LOG_LEVEL_TO_MUCH__, "BatchNorm2d.__init__: self.func_n", self.func_n)
        LOG(__LOG_LEVEL_TO_MUCH__, "BatchNorm2d.__init__: self.func_a", self.func_a)

        if out_quant_args == None:
            out_quant_args = (
                out_quant_bits,
                (-1,) if not out_quant_channel_wise else (1, num_features, 1, 1),
                0.1,
                "floor",
            )
        LOG(__LOG_LEVEL_TO_MUCH__, "BatchNorm2d.__init__: out_quant_args", out_quant_args)

        if out_quant == None:
            self.out_quant = LinQuantExpScale(*out_quant_args, **out_quant_kargs)
        else:
            self.out_quant = out_quant
        LOG(__LOG_LEVEL_TO_MUCH__, "BatchNorm2d.__init__: self.out_quant", self.out_quant)

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
            LOG(__LOG_LEVEL_HIGH_DETAIL__, "BatchNorm2d.get_weight_factor.ret_fun: self.alpha", self.alpha)
            return self.alpha[:, None, None, None]

        return ret_fun

    def forward(self, input: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x, rexp = input
        LOG(__LOG_LEVEL_HIGH_DETAIL__, "BatchNorm2d.forward: x", x)
        LOG(__LOG_LEVEL_HIGH_DETAIL__, "BatchNorm2d.forward: rexp", rexp)

        if self.training:
            if __HIGH_PRES__:
                xorig = x.data.clone().detach()

            x = super().forward(x)
            LOG(__LOG_LEVEL_TO_MUCH__, "BatchNorm2d.forward: x post super().forward", x)

            # remove this after mbv2
            # with torch.no_grad():
            #     nr = self.func_n(
            #         weight=torch.abs(self.weight.view(-1)),
            #         bias=self.bias.view(-1),
            #         mean=self.running_mean.view(-1),
            #         var=self.running_var.view(-1),
            #         out_quant=self.out_quant.delta_in.view(-1),
            #         rexp=rexp.view(-1),
            #     ).detach()
            #     LOG(__LOG_LEVEL_TO_MUCH__, "BatchNorm2d.forward: self.n", self.n)

            #     t = self.func_t(
            #         weight=self.weight.view(-1),
            #         bias=self.bias.view(-1),
            #         mean=0,
            #         var=self.running_var.view(-1),
            #         out_quant=self.out_quant.delta_in.view(-1),
            #         rexp=rexp.view(-1),
            #         n=self.n.view(-1),
            #     ).detach()

            #     tmp = torch.exp2(nr.view(1, -1, 1, 1))

            #     t = t.view(1, -1, 1, 1).div(tmp).floor().mul(tmp).mul(self.out_quant.delta_in.view(1,-1,1,1))
            #     x.data = x.data - (self.bias.view(1,-1,1,1)-t)  
            #upto here

            if not __HIGH_PRES__:
                LOG(__LOG_LEVEL_TO_MUCH__, "BatchNorm2d.forward: x post quant", x)
                x = self.out_quant(x, False)
            else:
                x = self.out_quant(x, True)
                with torch.no_grad():
                    if __HIGH_PRES_USE_RUNNING__:
                        mu = self.running_mean.clone()
                        var = self.running_var.clone()
                    else:
                        var = torch.var(xorig, [0, 2, 3], unbiased=False, keepdim=True)
                        mu = torch.mean(xorig, [0, 2, 3], keepdim=True)

                    n = self.weight.view(-1) / (
                        self.out_quant.delta_in.view(-1) * torch.sqrt(var.view(-1) + self.eps)
                    )
                    nr = self.func_n(
                        weight=torch.abs(self.weight.view(-1)),
                        bias=self.bias.view(-1),
                        mean=self.running_mean.view(-1),
                        var=self.running_var.view(-1),
                        out_quant=self.out_quant.delta_in.view(-1),
                        rexp=rexp.view(-1),
                    ).detach()
                    LOG(__LOG_LEVEL_TO_MUCH__, "BatchNorm2d.forward: self.n", self.n)

                    t = self.func_t(
                        weight=self.weight.view(-1),
                        bias=self.bias.view(-1),
                        mean=self.running_mean.view(-1),
                        var=self.running_var.view(-1),
                        out_quant=self.out_quant.delta_in.view(-1),
                        rexp=rexp.view(-1),
                        n=self.n.view(-1),
                    ).detach()

                    tmp = torch.exp2(nr.view(1, -1, 1, 1))

                    t = t.view(1, -1, 1, 1).div(tmp).floor().mul(tmp)

                    xorig = (
                        xorig
                        .mul_(n.view(1, -1, 1, 1))
                        .add_(t.view(1, -1, 1, 1))
                        .floor_()
                        .clamp_(min=self.out_quant.min, max=self.out_quant.max)
                        .mul_(self.out_quant.delta_out)
                    )
                x.data = xorig

            rexp = torch.log2(self.out_quant.delta_out)
            return x, rexp

        else:
            with torch.no_grad():
                self.n = self.func_n(
                    weight=torch.abs(self.weight.view(-1)),
                    bias=self.bias.view(-1),
                    mean=self.running_mean.view(-1),
                    var=self.running_var.view(-1),
                    out_quant=self.out_quant.delta_in.view(-1),
                    rexp=rexp.view(-1),
                ).detach()
                LOG(__LOG_LEVEL_TO_MUCH__, "BatchNorm2d.forward: self.n", self.n)

                t = self.func_t(
                    weight=self.weight.view(-1),
                    bias=self.bias.view(-1),
                    mean=self.running_mean.view(-1),
                    var=self.running_var.view(-1),
                    out_quant=self.out_quant.delta_in.view(-1),
                    rexp=rexp.view(-1),
                    n=self.n.view(-1),
                ).detach()
                LOG(__LOG_LEVEL_TO_MUCH__, "BatchNorm2d.forward: t", t)

                tmp = torch.exp2(self.n.view(1, -1, 1, 1))
                LOG(__LOG_LEVEL_TO_MUCH__, "BatchNorm2d.forward: tmp", tmp)

                self.t = t.view(1, -1, 1, 1).div(tmp).floor()
                LOG(__LOG_LEVEL_TO_MUCH__, "BatchNorm2d.forward: self.t", self.t)
                x = x + self.t
                LOG(__LOG_LEVEL_TO_MUCH__, "BatchNorm2d.forward: x post add", x)
                x = x.mul(tmp.view(1, -1, 1, 1))
                LOG(__LOG_LEVEL_TO_MUCH__, "BatchNorm2d.forward: x post shift", x)
                x = x.floor()
                LOG(__LOG_LEVEL_TO_MUCH__, "BatchNorm2d.forward: x post floor", x)
                x = x.clamp(self.out_quant.min, self.out_quant.max)
                LOG(__LOG_LEVEL_HIGH_DETAIL__, "BatchNorm2d.forward: x post clamp", x)
                if __DEBUG__:
                    if torch.any(torch.isnan(x)):
                        LOG(__LOG_LEVEL_IMPORTANT__, "BatchNorm2d.forward: nan in x", x)
                    x = torch.nan_to_num(x)
                    LOG(__LOG_LEVEL_TO_MUCH__, "BatchNorm2d.forward: x post nan to num", x)

                rexp = torch.log2(self.out_quant.delta_out)
                LOG(__LOG_LEVEL_HIGH_DETAIL__, "BatchNorm2d.forward: rexp out", rexp)
                return x, rexp
