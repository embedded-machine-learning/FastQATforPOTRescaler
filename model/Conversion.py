# Generic Type imports
from typing import Optional, Tuple, Union

# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import Tensor

# Self init
from . import logger_init, logger_forward
from .Quantizer import FakeQuant


class Start(nn.Module):
    """
    Start Transforms passed values into the quantized/fake quantized domain

    **IMPORTANT** A value domain of [-0.5,0.5] is assumed, fix this of different or force it to that domain

    :param bits: Quantization bit width
    :type bits: int
    """

    @logger_init
    def __init__(self, bits: int) -> None:
        """
        Please read Class help
        """
        # LOG(
        #     __LOG_LEVEL_DEBUG__,
        #     f"Start passed arguments:\n\
        #     bits:                           {bits}\n\
        #     ",
        # )
        super(Start, self).__init__()
        self.register_buffer("run", torch.tensor([-bits], dtype=torch.float))
        self.register_buffer("delta_in", torch.tensor([1.0 / (2.0 ** (-self.run) - 1)]))
        self.register_buffer("delta_out", torch.tensor([1.0 / (2.0 ** (-self.run))]))
        self.register_buffer("max", 2 ** (-self.run - 1) - 1)
        self.register_buffer("min", -(2 ** (-self.run - 1)))

    @logger_forward
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return (
            FakeQuant(
                x.clone(),
                self.delta_in,
                self.delta_out,
                self.training,
                self.min,
                self.max,
                "floor",
            ),
            self.run,
        )



class Stop(nn.Module):
    """
    Stop Return a Tensor pair from the fake-quantized/quantized domain
    """
    @logger_init
    def __init__(self, size=(1,)) -> None:
        """
        Please read Class help
        """
        super(Stop, self).__init__()
        self.size = size
        self.register_buffer("exp", torch.zeros(self.size))
        self.register_buffer("for_dtype", torch.zeros(1))  # Only required to know the current datatype
        
    @logger_forward
    def forward(self, invals: Tuple[Tensor, Tensor]) -> Tensor:
        x,rexp = invals
        self.exp = invals[1].detach().clone()
        # x = Stopfn.apply(invals[0], invals[1], self.training, self.for_dtype.dtype)
        with torch.no_grad():
            if not self.training:
                shape = [1 for _ in range(len(x.shape))]
                shape[1] = -1
                # LOG(__LOG_LEVEL_HIGH_DETAIL__, "Stopfn.forward: shape", shape)
                x = x.type(self.for_dtype.dtype).mul_(rexp.exp2().view(*shape))
                # LOG(__LOG_LEVEL_HIGH_DETAIL__, "Stopfn.forward: val", x)
        return x