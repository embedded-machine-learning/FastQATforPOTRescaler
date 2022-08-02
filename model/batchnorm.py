# Generic Type imports
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
        LOG(__LOG_LEVEL_DEBUG__, "calculate_n: n", n)
        n = n.ceil()
        LOG(__LOG_LEVEL_DEBUG__, "calculate_n: n post ceil", n)
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
        LOG(__LOG_LEVEL_DEBUG__, "calculate_n_fixed: n", n)
        nr = n.max() * torch.ones_like(n)
        LOG(__LOG_LEVEL_DEBUG__, "calculate_n_fixed: nr", nr)
        nr = nr.ceil()
        LOG(__LOG_LEVEL_DEBUG__, "calculate_n_fixed: nr post ceil", nr)
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
        t = -mean * (weight.abs() / (out_quant * torch.sqrt(var + 1e-5))) + bias / out_quant
        LOG(__LOG_LEVEL_DEBUG__, "calculate_t: t", t)
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
        n = torch.log2(weight.abs() * rexp.view(-1) / (out_quant * torch.sqrt(var + 1e-5)))
        LOG(__LOG_LEVEL_DEBUG__, "calculate_alpha: n", n)
        nr = torch.ceil(n)
        LOG(__LOG_LEVEL_DEBUG__, "calculate_alpha: nr", nr)
        alpha = torch.sign(weight)*torch.exp2(n - nr)
        LOG(__LOG_LEVEL_DEBUG__, "calculate_alpha: alpha", alpha)
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
        LOG(__LOG_LEVEL_DEBUG__, "calculate_alpha_fixed: n", n)
        nr = n.max() * torch.ones_like(n)
        LOG(__LOG_LEVEL_DEBUG__, "calculate_alpha_fixed: nr", nr)
        nr = torch.ceil(nr)
        LOG(__LOG_LEVEL_DEBUG__, "calculate_alpha_fixed: nr post ceil", nr)
        alpha = torch.sign(weight)*torch.exp2(n - nr)
        LOG(__LOG_LEVEL_DEBUG__, "calculate_alpha_fixed: alpha", alpha)
    raise NotImplementedError()
    return alpha


class BatchNorm2dBase(torch.nn.BatchNorm2d):
    """
    BatchNorm2dBase Modified nn.BatchNorm2d

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
    
    :param outQuantBits: Number of bits for the output quantization, defaults to 8
    :type outQuantBits: int, optional
    :param outQuantDyn: Channel-wise output quantization, defaults to False
    :type outQuantDyn: bool, optional
    :param fixed_n: Set the shift to a layer-wise value rather than channel-wise, defaults to False
    :type fixed_n: bool, optional
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
        outQuantBits: int = 8,
        outQuantDyn: bool = False,
        fixed_n: bool = False,
    ):
        """
        Please read the class help
        """
        LOG(__LOG_LEVEL_DEBUG__, f"BatchNorm2dBase passed arguments:\n\
        num_features:                {num_features}\n\
        eps:                         {eps}\n\
        momentum:                    {momentum}\n\
        affine:                      {affine}\n\
        track_running_stats:         {track_running_stats}\n\
        device:                      {device}\n\
        dtype:                       {dtype}\n\
        outQuantBits:                {outQuantBits}\n\
        outQuantDyn:                 {outQuantDyn}\n\
        fixed_n:                     {fixed_n}\n\
        ")
        
        super(BatchNorm2dBase, self).__init__(num_features, eps, momentum, affine, track_running_stats, device, dtype)

        self.register_buffer("n", torch.zeros(num_features))
        LOG(__LOG_LEVEL_TO_MUCH__, "BatchNorm2dBase: buffer n", self.n)
        self.register_buffer("t", torch.zeros(num_features))
        LOG(__LOG_LEVEL_TO_MUCH__, "BatchNorm2dBase: buffer t", self.t)
        self.register_buffer("alpha", 1.0 / np.sqrt(2.0) * torch.ones(num_features))
        LOG(__LOG_LEVEL_TO_MUCH__, "BatchNorm2dBase: buffer alpha", self.alpha)

        
        self.func_t = calculate_t
        self.fixed_n = fixed_n
        if fixed_n:
            self.func_n = calculate_n_fixed
            self.func_a = calculate_alpha_fixed
        else:
            self.func_n = calculate_n
            self.func_a = calculate_alpha
        LOG(__LOG_LEVEL_TO_MUCH__, "BatchNorm2dBase: self.func_n", self.func_n)
        LOG(__LOG_LEVEL_TO_MUCH__, "BatchNorm2dBase: self.func_a", self.func_a)
        

        self.out_quant = LinQuantExpScale(outQuantBits, (1, num_features, 1, 1) if outQuantDyn else (-1,), 1, "floor")
        LOG(__LOG_LEVEL_TO_MUCH__, "BatchNorm2dBase: self.out_quant", self.out_quant)
        

        self.register_buffer("weight_sign", torch.ones(num_features))
        LOG(__LOG_LEVEL_TO_MUCH__, "BatchNorm2dBase: buffer weight_sign", self.weight_sign)
        self.outQuantBits = outQuantBits


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
            LOG(__LOG_LEVEL_DEBUG__, "BatchNorm2dBase.get_weight_factor.ret_fun: self.alpha", self.alpha)
            return self.alpha[:, None, None, None]

        return ret_fun

    def forward(self, input: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x, rexp = input
        LOG(__LOG_LEVEL_DEBUG__,"BatchNorm2dBase.forward: x",x)
        LOG(__LOG_LEVEL_DEBUG__,"BatchNorm2dBase.forward: rexp",rexp)

        if self.training:
            if x.dtype == torch.int32:
                x = super().forward(x.type(self.weight.dtype))
            else:
                x = super().forward(x)
            LOG(__LOG_LEVEL_DEBUG__,"BatchNorm2dBase.forward: x post super().forward",x)

            x = self.out_quant(x)
            LOG(__LOG_LEVEL_DEBUG__,"BatchNorm2dBase.forward: x post quant",x)
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
                LOG(__LOG_LEVEL_DEBUG__,"BatchNorm2dBase.forward: self.n",self.n)

                t = self.func_t(
                    weight=self.weight.view(-1),
                    bias=self.bias.view(-1),
                    mean=self.running_mean.view(-1),
                    var=self.running_var.view(-1),
                    out_quant=self.out_quant.delta_in.view(-1),
                    rexp=rexp.view(-1),
                    n=self.n.view(-1),
                ).detach()
                LOG(__LOG_LEVEL_DEBUG__,"BatchNorm2dBase.forward: t",t)

                tmp = torch.exp2(self.n.type(torch.float32))
                LOG(__LOG_LEVEL_DEBUG__,"BatchNorm2dBase.forward: tmp",tmp)

                self.t = t.div(tmp).floor()
                LOG(__LOG_LEVEL_DEBUG__,"BatchNorm2dBase.forward: self.t",self.t)

                x = torch.nan_to_num(x)
                LOG(__LOG_LEVEL_DEBUG__,"BatchNorm2dBase.forward: x post nan to num",x)
                x = x + self.t[None, :, None, None]
                LOG(__LOG_LEVEL_DEBUG__,"BatchNorm2dBase.forward: x post add",x)
                x = x.mul(tmp[None, :, None, None])
                LOG(__LOG_LEVEL_DEBUG__,"BatchNorm2dBase.forward: x post shift",x)
                x = x.floor().type(torch.int32)
                LOG(__LOG_LEVEL_DEBUG__,"BatchNorm2dBase.forward: x post floor",x)
                x = x.clamp(self.out_quant.min,self.out_quant.max)
                LOG(__LOG_LEVEL_DEBUG__,"BatchNorm2dBase.forward: x post clamp",x)
                rexp = torch.log2(self.out_quant.delta_out)
                LOG(__LOG_LEVEL_DEBUG__,"BatchNorm2dBase.forward: rexp out",rexp)
                return x, rexp
