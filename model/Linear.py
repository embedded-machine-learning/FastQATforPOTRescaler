# Generic Type imports
from types import FunctionType
from typing import Union, Tuple

# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_2_t, Tensor

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


class LinQuantWeight(Quant):
    """
    LinQuantWeight Specialized Quantization for Linear Weights

    The special difference to the normal quantization methods is
        that it takes the exponent of the input and a function for
        the weight scaling, which does not modify the quantization
        factor.

    :param bits: Number of bits for quantization, defaults to 8
    :type bits: int, optional
    :param size: The shape of the quantization factor (-1,) would be 1 number (<channels>,1) would be a channel-wise quantization, defaults to (-1,)
    :type size: tuple, optional
    :param rounding_mode: Sets how the values are rounded `https://pytorch.org/docs/stable/generated/torch.div.html`, defaults to "trunc"
    :type rounding_mode: str, optional
    :param quant_int_dtype: The integer type used once quantized, defaults to torch.int32
    :type quant_int_dtype: torch.dtype, optional
    """

    def __init__(
        self,
        bits: int = 8,
        size: tuple = (-1,),
        rounding_mode: str = "trunc",
        quant_int_dtype: torch.dtype = torch.int32,
    ) -> None:
        """
        Please see class documentation `LinQuantWeight`
        """
        LOG(
            __LOG_LEVEL_HIGH_DETAIL__,
            f"LinQuantWeight passed arguments:\n\tsize:\t\t\t{size}\n\trounding_mode:\t{rounding_mode}\n\tquant_int_dtype:\t{quant_int_dtype}",
        )
        super(LinQuantWeight, self).__init__(size, rounding_mode, quant_int_dtype)

        self.bits = bits
        if size == (-1,):
            self.register_buffer("abs", torch.ones(1))
        else:
            self.register_buffer("abs", torch.ones(size))
        LOG(__LOG_LEVEL_TO_MUCH__, f"LinQuantWeight.__init__: abs buffer: {self.abs}")

        self.take_new = True

        assert self.bits > 0
        self.register_buffer("delta_in_factor", torch.tensor(2.0 / (2.0**self.bits)))
        LOG(__LOG_LEVEL_TO_MUCH__, f"LinQuantWeight.__init__: delta_in_factor buffer", self.delta_in_factor)
        self.register_buffer("delta_out_factor", torch.tensor(2.0 / (2.0**self.bits - 2)))
        LOG(__LOG_LEVEL_TO_MUCH__, f"LinQuantWeight.__init__: delta_out_factor buffer", self.delta_out_factor)

        self.register_buffer("max", torch.tensor(2 ** (self.bits - 1) - 1))
        LOG(__LOG_LEVEL_TO_MUCH__, f"LinQuantWeight.__init__: max buffer", self.max)
        self.register_buffer("min", torch.tensor(-(2 ** (self.bits - 1) - 1)))
        LOG(__LOG_LEVEL_TO_MUCH__, f"LinQuantWeight.__init__: min buffer", self.min)

    def forward(self, x: Tensor, rexp_mean: Tensor, rexp_diff: Tensor, fact_fun: FunctionType) -> tuple[Tensor, Tensor]:
        """
        forward Does the quantization, if :cvar:`self.training` returns floats else ints

        Calculates the quantization factors and a scaling factor defined by the passed function.

        :param x: The weights to quantize
        :type x: Tensor
        :param rexp_mean: Mean of the exponent
        :type rexp_mean: Tensor
        :param rexp_diff: Difference of the individual exponents compared to the mean
        :type rexp_diff: Tensor
        :param fact_fun: A function taking one value to calculate a scaling factor to force a following shift operation to a whole number
        :type fact_fun: FunctionType
        :return: Returns the Quantized weights and the scaling factor for debug purposes
        :rtype: tuple[Tensor,Tensor]
        """
        with torch.no_grad():
            abs = get_abs(self, x * (rexp_diff.view(1, -1)))
            LOG(__LOG_LEVEL_DEBUG__, f"LinQuantWeight.forward: abs", abs)

            self.abs = abs.detach()
            LOG(__LOG_LEVEL_TO_MUCH__, f"LinQuantWeight.forward: self.abs", self.abs)
            self.delta_in = self.abs.mul(self.delta_in_factor).detach()
            LOG(__LOG_LEVEL_DEBUG__, f"LinQuantWeight.forward: self.delta_in", self.delta_in)
            self.delta_out = self.abs.mul(self.delta_out_factor).detach()
            LOG(__LOG_LEVEL_DEBUG__, f"LinQuantWeight.forward: self.delta_out", self.delta_out)

            fact = fact_fun(self.delta_out * rexp_mean).view(-1, 1)
            LOG(__LOG_LEVEL_DEBUG__, f"LinQuantWeight.forward: fact", fact)

        return (
            FakeQuant(
                x=x.clone(),
                delta_in=self.delta_in / ((rexp_diff.view(1, -1) * fact)),
                delta_out=self.delta_out / ((rexp_diff.view(1, -1) * fact)),
                training=self.training,
                min_quant=self.min,
                max_quant=self.max,
                rounding_mode=self.rounding_mode,
                quant_int_dtype=self.quant_int_dtype,
            ),
            fact,
        )


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
    :param quant_int_dtype: The desired integer type, defaults to torch.int32
    :type quant_int_dtype: torch.dtype, optional
    :param quant_float_dtype: The desired floating-point type, defaults to torch.float32
    :type quant_float_dtype: torch.dtype, optional
    """

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
        out_quant_bits=8,
        out_quant_channel_wise=False,
        out_quant_args=None,
        out_quant_kargs={},
        quant_int_dtype: torch.dtype = torch.int32,
        quant_float_dtype: torch.dtype = torch.float32,
    ) -> None:
        LOG(
            __LOG_LEVEL_DEBUG__,
            f"Linear passed arguments:\n\
            in_features:                    {in_features}\n\
            out_features:                   {out_features}\n\
            bias:                           {bias}\n\
            device:                         {device}\n\
            dtype:                          {dtype}\n\
            weight_quant:                   {weight_quant}\n\
            weight_quant_bits:              {weight_quant_bits}\n\
            weight_quant_channel_wise:      {weight_quant_channel_wise}\n\
            weight_quant_args:              {weight_quant_args}\n\
            weight_quant_kargs:             {weight_quant_kargs}\n\
            out_quant:                      {out_quant}\n\
            out_quant_bits:                 {out_quant_bits}\n\
            out_quant_channel_wise:         {out_quant_channel_wise}\n\
            out_quant_args:                 {out_quant_args}\n\
            out_quant_kargs:                {out_quant_kargs}\n\
            quant_int_dtype:                {quant_int_dtype}\n\
            quant_float_dtype:              {quant_float_dtype}\n\
            ",
        )
        super().__init__(in_features, out_features, bias, device, dtype)

        if weight_quant_args == None:
            weight_quant_args = (
                weight_quant_bits,
                (-1,) if not weight_quant_channel_wise else (out_features, 1),
                "trunc",
            )
        LOG(__LOG_LEVEL_DEBUG__, "Linear.__init__: weight_quant_args", weight_quant_args)

        if weight_quant == None:
            self.weight_quant = LinQuantWeight(*weight_quant_args, **weight_quant_kargs)
        else:
            self.weight_quant = weight_quant
        LOG(__LOG_LEVEL_DEBUG__, "Linear.__init__: self.weight_quant", self.weight_quant)

        # only used if factor_fun in forward is None
        if out_quant_args == None:
            out_quant_args = (
                out_quant_bits,
                (-1,) if not out_quant_channel_wise else (1, out_features),
                0.1,
                "floor",
            )
        LOG(__LOG_LEVEL_DEBUG__, "Linear.__init__: out_quant_args", out_quant_args)

        if out_quant == None:
            self.out_quant = LinQuantExpScale(*out_quant_args, **out_quant_kargs)
        else:
            self.out_quant = out_quant
        LOG(__LOG_LEVEL_DEBUG__, "Linear.__init__: self.out_quant", self.out_quant)

        self.register_buffer("quant_weight", torch.zeros_like(self.weight))
        LOG(__LOG_LEVEL_TO_MUCH__, f"Linear.__init__: buffer quant_weight", self.quant_weight)
        self.register_buffer("n", torch.zeros((out_features if out_quant_channel_wise else 1)))
        LOG(__LOG_LEVEL_TO_MUCH__, f"Linear.__init__: buffer n", self.n)
        if bias:
            self.register_buffer("t", torch.zeros((out_features)))
        else:
            self.t = None
        LOG(__LOG_LEVEL_TO_MUCH__, f"Linear.__init__: buffer t", self.t)

        self.quant_int_dtype = quant_int_dtype
        LOG(__LOG_LEVEL_TO_MUCH__, f"Linear.__init__: self.quant_int_dtype", self.quant_int_dtype)
        self.quant_float_dtype = quant_float_dtype
        LOG(__LOG_LEVEL_TO_MUCH__, f"Linear.__init__: self.quant_float_dtype", self.quant_float_dtype)

    def get_weight_factor(self, delta_O: Tensor):
        """
        get_weight_factor returns a calculation function for the weight scaling

        :param delta_O: Output quantization factor
        :type delta_O: Tensor
        """

        def fun(rexp):
            with torch.no_grad():
                n = rexp.view(-1) / delta_O.view(-1)
                LOG(__LOG_LEVEL_TO_MUCH__, f"Linear.get_weight_factor.fun: pre ceil(log()) n", n)
                n = torch.log2(n)
                LOG(__LOG_LEVEL_TO_MUCH__, f"Linear.get_weight_factor.fun: pre ceil() n", n)
                nr = torch.ceil(n)
                LOG(__LOG_LEVEL_TO_MUCH__, f"Linear.get_weight_factor.fun: nr", nr)
                return torch.exp2(n - nr)

        LOG(__LOG_LEVEL_TO_MUCH__, f"Linear.get_weight_factor: fun", fun)
        return fun

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
            LOG(__LOG_LEVEL_TO_MUCH__, f"Linear.calculate_n: pre ceil(log()) n", n)
            n = torch.log2(n)
            LOG(__LOG_LEVEL_TO_MUCH__, f"Linear.calculate_n: pre ceil() n", n)
            if __DEBUG__:
                self.debug_n = n.clone()
            nr = torch.ceil(n)
            LOG(__LOG_LEVEL_DEBUG__, f"Linear.calculate_n: nr", nr)
        return nr

    def forward(self, invals: Tuple[torch.Tensor, torch.Tensor], factor_fun: FunctionType = None) -> torch.Tensor:
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
        input, rexp = invals
        LOG(__LOG_LEVEL_DEBUG__, "Linear.forward input", input)
        LOG(__LOG_LEVEL_DEBUG__, "Linear.forward rexp", rexp)

        rexp_mean = (torch.mean(rexp)).squeeze()
        LOG(__LOG_LEVEL_TO_MUCH__, "Linear.forward rexp_mean", rexp_mean)
        rexp_diff = rexp.squeeze() - rexp_mean.unsqueeze(-1)
        LOG(__LOG_LEVEL_TO_MUCH__, "Linear.forward rexp_diff", rexp_diff)

        weight = self.weight

        if factor_fun == None:
            weight, fact = self.weight_quant(
                weight.type(self.quant_float_dtype),
                rexp_mean.type(self.quant_float_dtype).exp2(),
                rexp_diff.type(self.quant_float_dtype).exp2(),
                self.get_weight_factor(self.out_quant.delta_in.view(-1).detach()),
            )
        else:
            weight, fact = self.weight_quant(
                weight.type(self.quant_float_dtype),
                rexp_mean.type(self.quant_float_dtype).exp2(),
                rexp_diff.type(self.quant_float_dtype).exp2(),
                factor_fun,
            )
        LOG(__LOG_LEVEL_DEBUG__, "Linear.forward weight", weight)
        LOG(__LOG_LEVEL_TO_MUCH__, "Linear.forward fact", fact)

        weight = weight.type(self.weight.dtype)
        fact = fact.type(self.weight.dtype)

        if self.bias == None:
            bias = None
        else:
            bias = FakeQuant(
                x=self.bias.clone().view(1,-1),
                delta_in=self.out_quant.delta_in,
                delta_out=self.out_quant.delta_out,
                training=self.training,
                min_quant=self.out_quant.min,
                max_quant=self.out_quant.max,
                rounding_mode=self.out_quant.rounding_mode,
                quant_int_dtype=self.out_quant.quant_int_dtype,
            )
        LOG(__LOG_LEVEL_DEBUG__, "Linear.forward bias", bias)

        if not self.training:
            if bias != None:
                self.t = bias.detach().view(1, -1)
            else:
                self.t = None
            LOG(__LOG_LEVEL_DEBUG__, "Linear.forward self.t", self.t)

            self.quant_weight = weight.detach()
            self.n = self.calculate_n(
                self.weight_quant.delta_out.view(-1).detach(),
                2 ** rexp_mean.view(-1).detach(),
                self.out_quant.delta_in.view(-1).detach(),
            ).view(1, -1)
            LOG(__LOG_LEVEL_DEBUG__, "Linear.forward self.n", self.n)

        # if torch.any(torch.isnan(weight)):
        #     print(torch.max(torch.abs(self.weight.view(-1))))

        # input = checkNan.apply( input, "Linear input")
        if self.training:
            out = F.linear(input, weight, bias)
        else:
            out = F.linear(
                input.type(self.quant_float_dtype),
                weight.type(self.quant_float_dtype),
                None,
            ).type(self.quant_int_dtype)
        LOG(__LOG_LEVEL_DEBUG__, "Linear.forward out", out)

        if factor_fun == None:
            if self.training:
                out2 = self.out_quant(out)
            else:
                if bias != None:
                    out2 = (
                        out.mul(torch.exp2(self.n)).add_(self.t).clamp_(self.out_quant.min, self.out_quant.max).floor_()
                    )
                else:
                    out2 = out.mul(torch.exp2(self.n)).clamp_(self.out_quant.min, self.out_quant.max).floor_()
            LOG(__LOG_LEVEL_DEBUG__, "Linear.forward out2", out2)
            return out2, torch.log2(self.out_quant.delta_out.detach()).view(1,-1)
        else:
            return out, rexp_mean + self.weight_quant.delta_out.log2().view(1,-1)
