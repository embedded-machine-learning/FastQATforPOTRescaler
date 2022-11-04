from types import FunctionType
from numpy import outer

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.common_types import Tensor

from .Linear_int import Linear_int

from ..logger import logger_forward, logger_init

from ..Quantizer import LinQuantExpScale, FakeQuant
from ..DataWrapper import DataWrapper
from .. import __DEBUG__

from .weight_quantization import LinQuantWeight


class Linear(nn.Linear):
    """
    Linear The Linear Class for a Quantized Dense Layer

    **IMPORTANT** Acts as an independent class if no function is passed to the forward method (if independent it quantizes the output by shifting)

    It is a linear Layer from `torch.nn.Linear` with modifications to allow for weight shaping

    :param in_features: Number of input features
    :type in_features: int
    :param out_features: Number of output features
    :type out_features: int
    :param bias: If True use a bias, defaults to True
    :type bias: bool, optional

    :param weight_quant: A callable object which overrides the default weight quantization, gets called with (weight,rexp_mean,rexp_diff,alpha_func(Tensor)->Tensor) , defaults to None
    :type weight_quant: class or function, optional
    :param weight_quant_bits: Number of bits , defaults to 8
    :type weight_quant_bits: int, optional
    :param weight_quant_channel_wise: If True makes a channel-wise quantization, defaults to False
    :type weight_quant_channel_wise: bool, optional
    :param weight_quant_args: Overrides arguments for the weight quantization initializer with custom ones, defaults to None
    :type weight_quant_args: _type_, optional
    :param weight_quant_kargs: Passes named arguments to the initializer of the weight quantization class, defaults to {}
    :type weight_quant_kargs: dict, optional

    :param out_quant: A callable object which overrides the default output quantization, gets called with (values) , defaults to None
    :type out_quant: _type_, optional
    :param out_quant_bits: Number of bits, defaults to 8
    :type out_quant_bits: int, optional
    :param out_quant_channel_wise: If True makes a channel-wise quantization, defaults to False
    :type out_quant_channel_wise: bool, optional
    :param out_quant_args: Overrides arguments for the out quantization initializer with custom ones, defaults to None
    :type out_quant_args: _type_, optional
    :param out_quant_kargs: Passes named arguments to the initializer of the out quantization class, defaults to {}
    :type out_quant_kargs: dict, optional
    """

    @logger_init
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        weight_quant=None,
        weight_quant_bits=8,
        weight_quant_channel_wise=False,
        weight_quant_args=None,
        weight_quant_kargs={},
        out_quant=None,
        out_quant_args=None,
        out_quant_kargs={},
    ) -> None:

        super(Linear, self).__init__(in_features, out_features, bias, device, dtype)

        if weight_quant_args == None:
            weight_quant_args = (
                weight_quant_bits,
                1 if not weight_quant_channel_wise else (out_features, 1),
                "trunc",
            )

        if weight_quant == None:
            self.weight_quant = LinQuantWeight(*weight_quant_args, **weight_quant_kargs)
        else:
            self.weight_quant = weight_quant

        # only used if factor_fun in forward is None
        if out_quant_args == None:
            out_quant_args = (
                8,
                (1, out_features),
            )

        if out_quant == None:
            self.out_quant = LinQuantExpScale(*out_quant_args, **out_quant_kargs)
        else:
            self.out_quant = out_quant(*out_quant_args, **out_quant_kargs)

        self.register_buffer("quant_weight", torch.zeros_like(self.weight))
        self.register_buffer("n", torch.zeros(((1, out_features))))
        if bias:
            self.register_buffer("t", torch.zeros((1, out_features)))
        else:
            self.t = None

    def int_extract(self, accumulation_type = torch.int32, small_signed_type = torch.int8, small_unsigned_type=torch.uint8) -> Linear_int:
        return Linear_int(
            self.in_features,
            self.out_features,
            self.quant_weight,
            self.n,
            self.t if self.bias is not None else None,
            self.out_quant.min,
            self.out_quant.max,
            accumulation_type = accumulation_type,
            small_signed_type = small_signed_type,
            small_unsigned_type = small_unsigned_type,
        )

    @logger_forward
    def get_weight_factor(self, delta_O: Tensor):
        """
        get_weight_factor returns a calculation function for the weight scaling

        :param delta_O: Output quantization factor
        :type delta_O: Tensor
        """

        def fun(rexp):
            with torch.no_grad():
                n = rexp.view(-1) / delta_O.view(-1)
                n = torch.log2(n)
                nr = torch.ceil(n)
                return torch.exp2(n - nr)

        return fun

    @logger_forward
    def calculate_n(self, delta_W: Tensor, delta_I: Tensor, delta_O: Tensor) -> Tensor:
        """
        calculate_n calculates the scaling shift

        :param delta_W: Weight scaling factor
        :type delta_W: Tensor
        :param delta_I: Input scaling factor
        :type delta_I: Tensor
        :param delta_O: Output scaling factor
        :type delta_O: Tensor
        :return: The shift value
        :rtype: Tensor
        """
        with torch.no_grad():
            n = delta_W.view(-1) * delta_I.view(-1) / delta_O.view(-1)
            n = torch.log2(n)
            if __DEBUG__:
                self.debug_n = n.clone()
            nr = torch.ceil(n)
        return nr

    @logger_forward
    def forward(self, invals: DataWrapper, factor_fun: FunctionType = None) -> torch.Tensor:
        """
        forward Computes the Linear layer with quantization

        **IMPORTANT** Acts as an independent class if no function is passed to the forward method (if independent it quantizes the output by shifting)

        :param invals: The values of the previous layer
        :type invals: Tuple[torch.Tensor, torch.Tensor]
        :param factor_fun: A function for additional weight scaling , defaults to None
        :type factor_fun: FunctionType, optional
        :return: Returns the computed values and the exponents
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        x, rexp = invals.get()

        rexp_mean = (torch.mean(rexp)).squeeze()
        rexp_diff = rexp.squeeze() - rexp_mean.unsqueeze(-1)

        weight = self.weight

        if factor_fun == None:
            weight, fact = self.weight_quant(
                weight,
                rexp_mean.exp2(),
                rexp_diff.exp2(),
                self.get_weight_factor(self.out_quant.delta_in.view(-1).detach()),
            )
        else:
            weight, fact = self.weight_quant(
                weight,
                rexp_mean.exp2(),
                rexp_diff.exp2(),
                factor_fun,
            )

        # weight = weight.type(self.weight.dtype)
        # fact = fact.type(self.weight.dtype)

        if self.bias == None:
            bias = None
        else:
            bias = FakeQuant(
                x=self.bias.clone().view(1, -1),
                delta_in=self.out_quant.delta_in,
                delta_out=self.out_quant.delta_out,
                training=self.training,
                min_quant=self.out_quant.min,
                max_quant=self.out_quant.max,
                rounding_mode=self.out_quant.rounding_mode,
            )

        if not self.training:
            if bias != None:
                self.t = bias.detach().view(1, -1)
            else:
                self.t = None

            self.quant_weight = weight.detach().clone()
            self.n = self.calculate_n(
                self.weight_quant.delta_out.view(-1).detach(),
                2 ** rexp_mean.view(-1).detach(),
                self.out_quant.delta_in.view(-1).detach(),
            ).view(1, -1)

        if self.training:
            out = F.linear(x, weight, bias)
        else:
            out = F.linear(x, weight, None)

        if factor_fun == None:
            if self.training:
                out2 = self.out_quant(out)
            else:
                if bias is not None:
                    out2 = (
                        out.mul(torch.exp2(self.n)).add_(self.t).clamp_(self.out_quant.min, self.out_quant.max).floor_()
                    )
                else:
                    out2 = out.mul(torch.exp2(self.n)).clamp_(self.out_quant.min, self.out_quant.max).floor_()
            return invals.set(out2, torch.log2(self.out_quant.delta_out.detach()))
        else:
            return invals.set(out, rexp_mean + self.weight_quant.delta_out.log2().view(1, -1))
