import torch
import torch.nn as nn


from ..logger import logger_forward,logger_init
from ..Quantizer import LinQuantExpScale 

from .. import __DEBUG__,__HIGH_PRES__

class Add(nn.Module):
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
    """
    @logger_init
    def __init__(
        self,
        size=(1,),
        out_quant=None,
        out_quant_bits: int = 8,
        out_quant_channel_wise: bool = False,
        out_quant_args=None,
        out_quant_kargs={},
    ) -> None:
        super(Add, self).__init__()

        self.register_buffer("a_shift", torch.zeros(size))
        self.register_buffer("b_shift", torch.zeros(size))

        if out_quant_args == None:
            out_quant_args = (
                out_quant_bits,
                (-1,) if not out_quant_channel_wise else size,
                0.1,
                "floor",
            )

        if out_quant == None:
            self.out_quant = LinQuantExpScale(*out_quant_args, **out_quant_kargs)
        else:
            self.out_quant = out_quant

    def forward(self, a, b):
        if a[0].shape != b[0].shape:
            raise torch.ErrorReport("testW")
        if self.training:
            out = a[0] + b[0]
            # checkNan.apply(a[0],"AddQAT a[0]")
            # checkNan.apply(b[0],"AddQAT b[0]")
            # out = checkNan.apply(out,"AddQAT out")

            out = self.out_quant(out, __HIGH_PRES__)
            # out = checkNan.apply(out,"AddQAT out post quant")
            rexp = self.out_quant.delta_out.log2()
            if __HIGH_PRES__:
                with torch.no_grad():
                    va = a[0].div(rexp.exp2()).floor()
                    vb = b[0].div(rexp.exp2()).floor()
                    out2 = va + vb
                    out2 = out2.clamp(self.out_quant.min, self.out_quant.max)
                    out2 = out2.mul(rexp.exp2())
                out.data = out2
        else:
            rexp = self.out_quant.delta_out.log2()
            self.a_shift = (rexp - a[1]).detach()
            self.b_shift = (rexp - b[1]).detach()
            va = a[0].div(self.a_shift.exp2(), rounding_mode="floor")
            vb = b[0].div(self.b_shift.exp2(), rounding_mode="floor")
            out = va + vb
            out = out.clamp(self.out_quant.min, self.out_quant.max)

        return out, rexp
