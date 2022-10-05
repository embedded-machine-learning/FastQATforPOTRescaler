# Generic Type imports
from types import FunctionType
from typing import Union, Tuple

# Torch imports
import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t, Tensor

# module imports
from ..Quantizer import Quant, get_abs, FakeQuant, LinQuantExpScale
from ..logger import logger_init,logger_forward
from ..Type import Data_wrapper

# current module imports
from .weight_quantization import LinQuantWeight


class Conv2d(nn.Conv2d):
    """
    Conv2dQuant The Convolution Class for a quantized convolution

    **IMPORTANT** Acts as an independent class if no function is passed to the forward method (if independent it quantizes the output by shifting)

    It is a 2d convolution from `torch.nn.Conv2d` with modifications to allow for weight shaping

    :param in_channels: Number of input channels
    :type in_channels: int
    :param out_channels: Number of output channels
    :type out_channels: int
    :param kernel_size: The kernal size
    :type kernel_size: _size_2_t
    :param stride: The stride, defaults to 1
    :type stride: _size_2_t, optional
    :param padding: The padding either a number od a string describing it, defaults to 0
    :type padding: Union[str, _size_2_t], optional
    :param dilation: Dilation identical to `torch.nn.Conv2d`, defaults to 1
    :type dilation: _size_2_t, optional
    :param groups: Groups, defaults to 1
    :type groups: int, optional
    :param bias: Adds a trainable bias if True, defaults to True
    :type bias: bool, optional
    :param padding_mode: padding mode identical to `torch.nn.Conv2d`, defaults to "zeros"
    :type padding_mode: str, optional

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
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
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
    ) -> None:
        super(Conv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )

        assert groups == 1 or (groups == in_channels and groups == out_channels)

        # Weight Quant
        if weight_quant_args == None:
            weight_quant_args = (
                weight_quant_bits,
                (-1,) if not weight_quant_channel_wise else (out_channels, 1, 1, 1),
                "trunc",
            )

        self.layer_wise = groups == in_channels

        if weight_quant == None:
            self.weight_quant = LinQuantWeight(*weight_quant_args, **weight_quant_kargs)
        else:
            self.weight_quant = weight_quant

        # Out Quant
        # only used if factor_fun in forward is None
        if out_quant_args == None:
            out_quant_args = (
                out_quant_bits,
                (-1,) if not out_quant_channel_wise else (1, out_channels, 1, 1),
                0.1,
                "floor",
            )

        if out_quant == None:
            self.out_quant = LinQuantExpScale(*out_quant_args, **out_quant_kargs)
        else:
            self.out_quant = out_quant

        self.register_buffer("quant_weight", torch.zeros_like(self.weight))
        self.register_buffer("n", torch.zeros(((1, out_channels, 1, 1) if out_quant_channel_wise else 1)))
        if bias:
            self.register_buffer("t", torch.zeros(((1, out_channels, 1, 1) if out_quant_channel_wise else 1)))
        else:
            self.t = None

        # if __DEBUG__:
        #     self.debug_fact = []
        #     self.debug_weight = []
        #     self.debug_bias = []
        #     self.debug_n = []

        self.register_buffer("rexp_diff", torch.zeros((1, in_channels, 1, 1) if self.layer_wise else (in_channels)))

        sh = self.weight.shape
        self.sh_prod = sh[1] * sh[2] * sh[3]

    @logger_forward
    def get_weight_factor(self, delta_O: Tensor):
        """
        get_weight_factor returns a calculation function for the weight scaling

        :param delta_O: Output quantization factor
        :type delta_O: Tensor
        """

        def fun(rexp):
            with torch.no_grad():
                # print(delta_I,delta_O,delta_W)
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
            nr = torch.ceil(n)
        return nr

    @logger_forward
    def forward(
        self, invals: Data_wrapper, factor_fun: FunctionType = None
    ) -> Data_wrapper:
        """
        forward Computes the convolution with quantized weights

        **IMPORTANT** Acts as an independent class if no function is passed to the forward method (if independent it quantizes the output by shifting)

        :param invals: The values of the previous layer
        :type invals: Tuple[torch.Tensor, torch.Tensor]
        :param factor_fun: A function for additional weight scaling , defaults to None
        :type factor_fun: FunctionType, optional
        :return: Returns the computed values and the exponents
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        input, rexp = invals.get()

        # Questinable follows
        # if self.training and self.layer_wise:
        # # if self.training :
        #     mean = self.weight.data.mean((1,2,3),keepdim=True)
        #     var  = self.weight.data.var((1,2,3),keepdim=True).add(1e-5).sqrt()
        #     mod = (mean.sign())*((torch.abs(mean)-var).clamp(min=0))
        #     self.weight.data = self.weight.data - mod

        # if self.training:
        #     var  = self.weight.data.var((1,2,3),keepdim=True).add(1e-5).sqrt()
        #     self.weight.data = self.weight.data/(math.sqrt(self.sh_prod)*var)
        # Done

        if self.layer_wise:
            rexp_mean = rexp.clone().detach()
            self.rexp_diff = rexp - rexp_mean
        else:
            rexp_mean = (torch.mean(rexp)).squeeze()
            self.rexp_diff = rexp.squeeze() - rexp_mean.unsqueeze(-1)

        weight = self.weight.clone()

        if factor_fun == None:
            weight, fact = self.weight_quant(
                weight,
                rexp_mean.exp2(),
                self.rexp_diff.exp2(),
                self.get_weight_factor(self.out_quant.delta_in.view(-1).detach()),
            )
        else:
            weight, fact = self.weight_quant(
                weight,
                rexp_mean.exp2(),
                self.rexp_diff.exp2(),
                factor_fun,
            )

        # weight = weight.type(self.weight.dtype)
        # fact = fact.type(self.weight.dtype)

        if self.bias == None:
            bias = None
        else:
            bias = FakeQuant(
                x=self.bias.clone(),
                delta_in=self.out_quant.delta_in.view(-1),
                delta_out=self.out_quant.delta_out.view(-1),
                training=self.training,
                min_quant=self.out_quant.min.view(-1),
                max_quant=self.out_quant.max.view(-1),
                rounding_mode=self.out_quant.rounding_mode,
            )

        if not self.training:
            if bias != None:
                self.t = bias.detach().view(1, -1, 1, 1)
            else:
                self.t = None

            self.quant_weight = weight.detach()
            if factor_fun == None:
                self.n = self.calculate_n(
                    self.weight_quant.delta_out.view(-1).detach(),
                    2 ** rexp_mean.view(-1).detach(),
                    self.out_quant.delta_in.view(-1).detach(),
                ).view(1, -1, 1, 1)

        # if __DEBUG__:
        #     self.debug_fact = fact
        #     self.debug_weight = weight
        #     self.debug_bias = bias
        #     self.debug_n = self.n.clone()

        if self.training:
            out = self._conv_forward(input, weight, bias)
        else:
            out = self._conv_forward(
                input,
                weight,
                None,
            )

        if factor_fun == None:
            if self.training:
                out2 = self.out_quant(out)
            else:
                if bias != None:
                    out2 = (
                        out.mul(torch.exp2(self.n)).floor_().add_(self.t).clamp_(self.out_quant.min, self.out_quant.max)
                    )
                else:
                    out2 = out.mul(torch.exp2(self.n)).floor_().clamp_(self.out_quant.min, self.out_quant.max)
            return invals.set(out2, torch.log2(self.out_quant.delta_out.detach()))
        else:
            return invals.set(out, rexp_mean + (self.weight_quant.delta_out.log2().detach().view(1, -1, 1, 1)))
