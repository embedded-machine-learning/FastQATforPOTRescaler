# Generic Type imports
import imp
from typing import Optional, Tuple, Union

# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_any_t, _size_2_t, Tensor, _size_any_opt_t

# Numpy
import numpy as np

# current module imports
from .quantizer import FakeQuant, LinQuantExpScale
from .convolution import Conv2d
from .batchnorm import BatchNorm2d
from .activations import LeakReLU

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

#########################################################################################
#                                   FUNCTIONS                                           #
#########################################################################################
class Stopfn(torch.autograd.Function):
    """
    Stopfn Takes a tensor and transforms it into `dtype` if not training

    :param val: The Tensor to be transformed
    :type val: Tensor
    :param rexp: The exponent
    :type rexp: Tensor
    :param training: If training is True nothing happens
    :type training: bool
    :param dtype: the desired datatype
    :type dtype: torch.dtype
    :return: The possibly transformed tensor
    :rtype: Tensor
    """

    @staticmethod
    def forward(self, val: Tensor, rexp: Tensor, training: bool, dtype: torch.dtype) -> Tensor:
        """
        Please read the help for the Class
        """
        with torch.no_grad():
            if not training:
                shape = [1 for _ in range(len(val.shape))]
                shape[1] = -1
                LOG(__LOG_LEVEL_HIGH_DETAIL__, "Stopfn.forward: shape", shape)
                val = val.type(dtype).mul_(rexp.exp2().view(*shape))
                LOG(__LOG_LEVEL_HIGH_DETAIL__, "Stopfn.forward: val", val)
        return val

    @staticmethod
    def backward(self, x: Tensor):
        return x, None, None, None


#########################################################################################
#                                   CLASSES                                             #
#########################################################################################


class Start(nn.Module):
    """
    Start Transforms passed values into the quantized/fake quantized domain

    **IMPORTANT** A value domain of [-0.5,0.5] is assumed, fix this of different or force it to that domain

    :param bits: Quantization bit width
    :type bits: int
    """

    def __init__(self, bits: int) -> None:
        """
        Please read Class help
        """
        LOG(
            __LOG_LEVEL_DEBUG__,
            f"Start passed arguments:\n\
            bits:                           {bits}\n\
            ",
        )
        super(Start, self).__init__()
        self.register_buffer("run", torch.tensor([-bits], dtype=torch.float))
        LOG(__LOG_LEVEL_TO_MUCH__, "Start.__init__: buffer run", self.run)
        self.register_buffer("delta_in", torch.tensor([1.0 / (2.0 ** (-self.run) - 1)]))
        LOG(__LOG_LEVEL_TO_MUCH__, "Start.__init__: buffer delta_in", self.delta_in)
        self.register_buffer("delta_out", torch.tensor([1.0 / (2.0 ** (-self.run))]))
        LOG(__LOG_LEVEL_TO_MUCH__, "Start.__init__: buffer delta_out", self.delta_out)
        self.register_buffer("max", 2 ** (-self.run - 1) - 1)
        LOG(__LOG_LEVEL_TO_MUCH__, "Start.__init__: buffer max", self.max)
        self.register_buffer("min", -(2 ** (-self.run - 1)))
        LOG(__LOG_LEVEL_TO_MUCH__, "Start.__init__: buffer min", self.min)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        LOG(__LOG_LEVEL_HIGH_DETAIL__, "Start.forward: x", x)
        x = FakeQuant(
            x.clone(),
            self.delta_in,
            self.delta_out,
            self.training,
            self.min,
            self.max,
            "floor",
            torch.int32,
        )
        return x, self.run


class Stop(nn.Module):
    """
    Stop Return a Tensor pair from the fake-quantized/quantized domain
    """

    def __init__(self,size=(1,)) -> None:
        """
        Please read Class help
        """
        super(Stop, self).__init__()
        self.size = size
        self.register_buffer("exp", torch.zeros(self.size))
        LOG(__LOG_LEVEL_TO_MUCH__, "Stop.__init: buffer exp", self.exp)
        self.register_buffer("for_dtype", torch.zeros(1))  # Only required to know the current datatype
        LOG(__LOG_LEVEL_TO_MUCH__, "Stop.__init: buffer for_dtype", self.for_dtype)

    def forward(self, invals: Tuple[Tensor, Tensor]) -> Tensor:
        self.exp = invals[1].detach().clone()
        x = Stopfn.apply(invals[0], invals[1], self.training, self.for_dtype.dtype)
        return x


#########################################################################################
#                                   BLOCKS                                              #
#########################################################################################


class BlockQuantN(nn.Module):
    """
    BlockQuantN A module with a Convolution BN and activation

    Per default the activation function is a leaky ReLu

    :param in_channels: Number of input channels
    :type in_channels: int
    :param out_channels: Number of output channels
    :type out_channels: int
    :param kernel_size: Kernel size for the Convolution
    :type kernel_size: _size_2_t
    :param stride: Stride for the Convolution, defaults to 1
    :type stride: _size_2_t, optional
    :param padding: padding for the Convolution, defaults to 0
    :type padding: Union[str, _size_2_t], optional
    :param dilation: Dilation for the Convolution, defaults to 1
    :type dilation: _size_2_t, optional
    :param groups: Groups for the Convolution, defaults to 1
    :type groups: int, optional
    :param padding_mode: Padding mode for the Convolution, defaults to "zeros"
    :type padding_mode: str, optional
    :param weight_quant: Overrides the default weight quantization for the Convolution, defaults to None
    :type weight_quant: _type_, optional
    :param weight_quant_bits: Number of bits for the Convolution Weight quantization, defaults to 8
    :type weight_quant_bits: int, optional
    :param weight_quant_channel_wise: If the Convolution Weight quantization should be done Layer-wise, defaults to False
    :type weight_quant_channel_wise: bool, optional
    :param weight_quant_args: Overrides the args for the Convolution Weight quantization, defaults to None
    :type weight_quant_args: _type_, optional
    :param weight_quant_kargs: Additional Named Arguments for the Convolution Weight quantization, defaults to {}
    :type weight_quant_kargs: dict, optional
    :param eps: EPS for the Batch-Norm , defaults to 1e-5
    :type eps: float, optional
    :param momentum: Momentum for the Batch-Norm, defaults to 0.1
    :type momentum: float, optional
    :param affine: Affine for the Batch-Norm, defaults to True
    :type affine: bool, optional
    :param track_running_stats: Trach running stats for the Batch-Norm, defaults to True
    :type track_running_stats: bool, optional
    :param fixed_n: If the batch-Norm should a single shift factor per layer, defaults to False
    :type fixed_n: bool, optional
    :param out_quant: Overrides the output quantization of the Batch-Norm, defaults to None
    :type out_quant: _type_, optional
    :param out_quant_bits: Number of bits for the output quantization of the Batch-Norm, defaults to 8
    :type out_quant_bits: int, optional
    :param out_quant_channel_wise: If the Batch-Norm output quantization should be done Channel-wise, defaults to False
    :type out_quant_channel_wise: bool, optional
    :param out_quant_args: Overrides the arguments for the batch-Norm output quantization, defaults to None
    :type out_quant_args: _type_, optional
    :param out_quant_kargs: Additional Named Arguments for the Batch-Norm output quantization, defaults to {}
    :type out_quant_kargs: dict, optional
    :param leaky_relu_slope: The LeakyRelu negative slope, defaults to 2**-6
    :type leaky_relu_slope: float, optional
    :param leaky_relu_inplace: If the Leaky Relu should be done inplace, defaults to False
    :type leaky_relu_inplace: bool, optional
    :param activation: Overrides the default activation function, e.g. nn.Sequential() for no activation, defaults to None
    :type activation: _type_, optional
    :param activation_args: Overrides the Arguments provided to the activation function, defaults to None
    :type activation_args: _type_, optional
    :param activation_kargs: Additional Named parameters for the activation function, defaults to {}
    :type activation_kargs: dict, optional
    :param quant_int_dtype: The desired integer type, defaults to torch.int32
    :type quant_int_dtype: torch.dtype, optional
    :param quant_float_dtype: The desired float type, defaults to torch.float32
    :type quant_float_dtype: torch.dtype, optional
    """

    def __init__(
        self,
        # Convolution
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        padding_mode: str = "zeros",
        weight_quant=None,
        weight_quant_bits=8,
        weight_quant_channel_wise=False,
        weight_quant_args=None,
        weight_quant_kargs={},
        # Batch-Norm
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        fixed_n: bool = False,
        out_quant=None,
        out_quant_bits=8,
        out_quant_channel_wise=False,
        out_quant_args=None,
        out_quant_kargs={},
        # Activation
        leaky_relu_slope: float = 2**-6,
        leaky_relu_inplace: bool = False,
        activation=None,
        activation_args=None,
        activation_kargs={},
        # General stuff
        quant_int_dtype: torch.dtype = torch.int32,
        quant_float_dtype: torch.dtype = torch.float32,
        device=None,
        dtype=None,
    ) -> None:
        """
        Please see class documentation
        """
        super(BlockQuantN, self).__init__()
        self.conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
            weight_quant=weight_quant,
            weight_quant_bits=weight_quant_bits,
            weight_quant_channel_wise=weight_quant_channel_wise,
            weight_quant_args=weight_quant_args,
            weight_quant_kargs=weight_quant_kargs,
            out_quant=None,
            out_quant_bits=1,
            out_quant_channel_wise=False,
            out_quant_args=None,
            out_quant_kargs={},
            quant_int_dtype=quant_int_dtype,
            quant_float_dtype=quant_float_dtype,
        )
        self.bn = BatchNorm2d(
            num_features=out_channels,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            device=device,
            dtype=dtype,
            fixed_n=fixed_n,
            out_quant=out_quant,
            out_quant_bits=out_quant_bits,
            out_quant_channel_wise=out_quant_channel_wise,
            out_quant_args=out_quant_args,
            out_quant_kargs=out_quant_kargs,
            quant_int_dtype=quant_int_dtype,
            quant_float_dtype=quant_float_dtype,
        )

        if activation_args == None:
            activation_args = (leaky_relu_slope, leaky_relu_inplace)

        if activation == None:
            self.activation = LeakReLU(*activation_args, **activation_kargs)
        else:
            self.activation = activation

    def forward(self, invals: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:

        fact = self.bn.get_weight_factor()

        x = self.conv(invals, fact)
        x = self.bn(x)
        x = self.activation(x)

        return x


class BlockQuantNwoA(BlockQuantN):
    """
    BlockQuantNwoA BlockQuantN with out Activation

    Please see `BlockQuantN` for details
    """

    def __init__(
        self,
        # Convolution
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        padding_mode: str = "zeros",
        weight_quant=None,
        weight_quant_bits=8,
        weight_quant_channel_wise=False,
        weight_quant_args=None,
        weight_quant_kargs={},
        # Batch-Norm
        eps: float = 0.00001,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        fixed_n: bool = False,
        out_quant=None,
        out_quant_bits=8,
        out_quant_channel_wise=False,
        out_quant_args=None,
        out_quant_kargs={},
        # General Stuff
        quant_int_dtype: torch.dtype = torch.int32,
        quant_float_dtype: torch.dtype = torch.float32,
        device=None,
        dtype=None,
    ) -> None:
        """
        Please see class documentation
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            padding_mode=padding_mode,
            weight_quant=weight_quant,
            weight_quant_bits=weight_quant_bits,
            weight_quant_channel_wise=weight_quant_channel_wise,
            weight_quant_args=weight_quant_args,
            weight_quant_kargs=weight_quant_kargs,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            fixed_n=fixed_n,
            out_quant=out_quant,
            out_quant_bits=out_quant_bits,
            out_quant_channel_wise=out_quant_channel_wise,
            out_quant_args=out_quant_args,
            out_quant_kargs=out_quant_kargs,
            activation=nn.Sequential(),
            quant_int_dtype=quant_int_dtype,
            quant_float_dtype=quant_float_dtype,
            device=device,
            dtype=dtype,
        )


#########################################################################################
#                                   Common Layers                                       #
#########################################################################################
class AddQAT(nn.Module):
    """
    AddQAT Adds 2 numbers

    there is an internal scaling and the required shift operations are being calculated

    :param num_features: number of features
    :type num_features: int
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
    :param quant_int_dtype: The desired integer type, defaults to torch.int32
    :type quant_int_dtype: torch.dtype, optional
    """

    def __init__(
        self,
        size=(1,),
        out_quant=None,
        out_quant_bits: int = 8,
        out_quant_channel_wise: bool = False,
        out_quant_args=None,
        out_quant_kargs={},
        quant_int_dtype: torch.dtype = torch.int32,
    ) -> None:
        LOG(
            __LOG_LEVEL_DEBUG__,
            f"AddQAT passed arguments:\n\
            size:                           {size}\n\
            out_quant:                      {out_quant}\n\
            out_quant_bits:                 {out_quant_bits}\n\
            out_quant_channel_wise:         {out_quant_channel_wise}\n\
            out_quant_args:                 {out_quant_args}\n\
            out_quant_kargs:                {out_quant_kargs}\n\
            quant_int_dtype:                {quant_int_dtype}\n\
            ",
        )
        super(AddQAT, self).__init__()

        self.register_buffer("a_shift", torch.zeros(size))
        LOG(__LOG_LEVEL_TO_MUCH__, "AddQAT.__init__: buffer a_shift", self.a_shift)
        self.register_buffer("b_shift", torch.zeros(size))
        LOG(__LOG_LEVEL_TO_MUCH__, "AddQAT.__init__: buffer b_shift", self.b_shift)

        if out_quant_args == None:
            out_quant_args = (
                out_quant_bits,
                (-1,) if not out_quant_channel_wise else size,
                0.1,
                "floor",
                quant_int_dtype,
            )
        LOG(__LOG_LEVEL_TO_MUCH__, "AddQAT.__init__: out_quant_args", out_quant_args)

        if out_quant == None:
            self.out_quant = LinQuantExpScale(*out_quant_args, **out_quant_kargs)
        else:
            self.out_quant = out_quant
        LOG(__LOG_LEVEL_TO_MUCH__, "AddQAT.__init__: self.out_quant", self.out_quant)

    def forward(self, a, b):
        if a[0].shape != b[0].shape:
            raise torch.ErrorReport("testW")
        if self.training:
            out = a[0] + b[0]
            LOG(__LOG_LEVEL_TO_MUCH__, "AddQAT.forward: out", out)
            out = self.out_quant(out.clone())
            LOG(__LOG_LEVEL_TO_MUCH__, "AddQAT.forward: out post quant", out)
            rexp = self.out_quant.delta_out.log2()
            LOG(__LOG_LEVEL_TO_MUCH__, "AddQAT.forward: rexp", rexp)
        else:
            rexp = self.out_quant.delta_out.log2()
            LOG(__LOG_LEVEL_TO_MUCH__, "AddQAT.forward: rexp", rexp)
            self.a_shift = -(a[1] - rexp).detach()
            LOG(__LOG_LEVEL_TO_MUCH__, "AddQAT.forward: self.a_shift", self.a_shift)
            self.b_shift = -(b[1] - rexp).detach()
            LOG(__LOG_LEVEL_TO_MUCH__, "AddQAT.forward: self.b_shift", self.b_shift)
            va = a[0].div(self.a_shift.exp2(), rounding_mode="floor")
            LOG(__LOG_LEVEL_TO_MUCH__, "AddQAT.forward: va", va)
            vb = b[0].div(self.b_shift.exp2(), rounding_mode="floor")
            LOG(__LOG_LEVEL_TO_MUCH__, "AddQAT.forward: vb", vb)
            out = va + vb
            LOG(__LOG_LEVEL_TO_MUCH__, "AddQAT.forward: out", out)
            out = out.clamp(self.out_quant.min, self.out_quant.max)
            LOG(__LOG_LEVEL_TO_MUCH__, "AddQAT.forward: out post clamp", out)

        return out, rexp


def Flatten(input: Tuple[torch.Tensor, torch.Tensor], dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Flatten encapsulation of torch.flatten
    """
    val, rexp = input
    orexp = rexp.detach() * torch.ones_like(val[0, :])
    return val.flatten(dim), orexp.flatten(dim)


#########################################################################################
#                                   ENCAPSULATED                                        #
#########################################################################################


class MaxPool2d(nn.MaxPool2d):
    def __init__(
        self,
        kernel_size: _size_any_t,
        stride: Optional[_size_any_t] = None,
        padding: _size_any_t = 0,
        dilation: _size_any_t = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ) -> None:
        super(MaxPool2d, self).__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)

    def forward(self, input: Tuple[torch.Tensor, torch.Tensor]):
        val, rexp = input
        if self.training:
            return (
                F.max_pool2d(
                    val, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode, self.return_indices
                ),
                rexp,
            )
        else:
            return (
                F.max_pool2d(
                    val.type(torch.float32),
                    self.kernel_size,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.ceil_mode,
                    self.return_indices,
                ).type(torch.int32),
                rexp,
            )


class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    def __init__(self, output_size: _size_any_opt_t) -> None:
        super().__init__(output_size)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # does nopt modify the channels so simple wrapping and floor should be enough
        val, rexp = x

        val = super().forward(val)

        if self.training:
            with torch.no_grad():
                val.data = val.data / torch.exp2(rexp.view(-1)[None, :, None, None])
                val.data = val.data.floor()
                val.data = val.data * torch.exp2(rexp.view(-1)[None, :, None, None])

        else:
            val = val.floor()

        return val, rexp
