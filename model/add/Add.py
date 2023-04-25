from typing import Union, Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn

from ..DataWrapper import DataWrapper

from ..logger import logger_forward, logger_init
from ..Quantizer import LinQuantExpScale, Quant

from .. import __HIGH_PRES__


class Add(nn.Module):
    """
    AddQAT Adds 2 numbers

    there is an internal scaling and the required shift operations are being calculated

    :param num_features: number of features
    :type num_features: int
    :param out_quant:  A callable object which overrides the default output quantization, gets called with (values) , defaults to None
    :type out_quant: _type_, optional
    :param out_quant_args:  Overrides arguments for the out quantization initializer with custom ones, defaults to None
    :type out_quant_args: _type_, optional
    :param out_quant_kargs: Passes named arguments to the initializer of the out quantization class, defaults to {}
    :type out_quant_kargs: dict, optional
    """

    @logger_init
    def __init__(
        self,
        size=(1,),
        out_quant=None,
        out_quant_args=None,
        out_quant_kargs={},
    ) -> None:
        super(Add, self).__init__()

        self.register_buffer("a_shift", torch.zeros(size))
        self.register_buffer("b_shift", torch.zeros(size))

        if out_quant_args == None:
            out_quant_args = (
                8,
                size,
            )

        if out_quant == None:
            self.out_quant = LinQuantExpScale(*out_quant_args, **out_quant_kargs)
        else:
            self.out_quant = out_quant(*out_quant_args, **out_quant_kargs)

    @logger_forward
    def forward(self, in_a: DataWrapper, in_b: DataWrapper, activation: Union[None, nn.Module] = None) -> DataWrapper:
        a = in_a.get()
        b = in_b.get()

        if activation != None:
            self.out_quant.copy(activation)
            quant = activation
        else:
            quant = self.out_quant

        if a[0].shape != b[0].shape:
            raise torch.ErrorReport("ADD: input shapes not identical", a[0].shape, b[0].shape)
        if self.training:
            out = a[0] + b[0]
            out = quant(out, __HIGH_PRES__, in_a)
            rexp = quant.delta_out.log2()
            if __HIGH_PRES__:
                with torch.no_grad():
                    va = a[0].div(rexp.exp2()).floor()
                    vb = b[0].div(rexp.exp2()).floor()
                    out2 = va + vb
                    out2 = out2.clamp(quant.min, quant.max)
                    out2 = out2.mul(rexp.exp2())
                out.data = out2
        else:
            rexp = quant.delta_out.log2()
            self.a_shift = (a[1] - rexp).detach()
            self.b_shift = (b[1] - rexp).detach()
            va = a[0].mul(self.a_shift.exp2()).floor()
            vb = b[0].mul(self.b_shift.exp2()).floor()
            out = va + vb
            out = out.clamp(quant.min, quant.max)

        return in_a.set(out, rexp)










class Hidden_ReLU(Quant):
    """
    ReLU The implementation of the ReLU activation function fused into the quantization

    :param size: The shape for alpha, defaults to (1,)
    :type size: tuple, optional
    """

    def __init__(self, bits, size=(-1,), rounding_mode: str = "floor", use_enforced_quant_level: bool = False, mom1: int  = 0.1) -> None:
        super(Hidden_ReLU, self).__init__(bits, size, rounding_mode, use_enforced_quant_level)
        self.bits = bits

        nn.init.constant_(self.min, 0)
        nn.init.constant_(self.max, 2**bits - 1)

    def set_quant(self, value1, value2):
        with torch.no_grad():
            exp1,exp2 = value1.exp2(), value2.exp2()
            exp = exp1 + exp2
            exp = exp.div(exp1).log2().sub(1).round().exp2().mul(exp1)
            self.delta_in = exp
            self.delta_out = exp

    def forward(self, x: torch.Tensor, fake: bool = False, metadata: Optional[DataWrapper] = None):
        x = RELU_back_function.apply(x)
        return super(Hidden_ReLU,self).forward(x, fake)


class RELU_back_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, val: Tensor) -> Tensor:
        ctx.save_for_backward(val > 0)
        return val.clone()

    @staticmethod
    def backward(ctx, grad_outputs: Tensor) -> Tuple[Tensor, Tensor]:
        (zero_cmp,) = ctx.saved_tensors
        val_gard = grad_outputs * zero_cmp
        return val_gard



class AddRELU(nn.Module):
    """
    AddQAT Adds 2 numbers

    there is an internal scaling and the required shift operations are being calculated

    :param num_features: number of features
    :type num_features: int
    :param out_quant:  A callable object which overrides the default output quantization, gets called with (values) , defaults to None
    :type out_quant: _type_, optional
    :param out_quant_args:  Overrides arguments for the out quantization initializer with custom ones, defaults to None
    :type out_quant_args: _type_, optional
    :param out_quant_kargs: Passes named arguments to the initializer of the out quantization class, defaults to {}
    :type out_quant_kargs: dict, optional
    """

    @logger_init
    def __init__(
        self,
        size=(1,),
        out_quant=None,
        out_quant_args=None,
        out_quant_kargs={},
    ) -> None:
        super(AddRELU, self).__init__()

        self.register_buffer("a_shift", torch.zeros(size))
        self.register_buffer("b_shift", torch.zeros(size))

        if out_quant_args == None:
            out_quant_args = (
                8,
                size,
            )

        self.out_quant = Hidden_ReLU(*out_quant_args, **out_quant_kargs)
        
    @logger_forward
    def forward(self, in_a: DataWrapper, in_b: DataWrapper, activation: Union[None, nn.Module] = None) -> DataWrapper:
        a = in_a.get()
        b = in_b.get()

        quant = self.out_quant

        if a[0].shape != b[0].shape:
            raise torch.ErrorReport("ADD: input shapes not identical", a[0].shape, b[0].shape)
        if self.training:
            out = a[0] + b[0]
            quant.set_quant(a[1], b[1])
            out = quant(out, __HIGH_PRES__, in_a)
            rexp = quant.delta_out.log2()
            if __HIGH_PRES__:
                with torch.no_grad():
                    va = a[0].div(rexp.exp2()).floor()
                    vb = b[0].div(rexp.exp2()).floor()
                    out2 = va + vb
                    out2 = out2.clamp(quant.min, quant.max)
                    out2 = out2.mul(rexp.exp2())
                out.data = out2
        else:
            rexp = quant.delta_out.log2()
            self.a_shift = (a[1] - rexp).detach()
            self.b_shift = (b[1] - rexp).detach()
            va = a[0].mul(self.a_shift.exp2()).floor()
            vb = b[0].mul(self.b_shift.exp2()).floor()
            out = va + vb
            out = out.clamp(quant.min, quant.max)

        return in_a.set(out, rexp)
