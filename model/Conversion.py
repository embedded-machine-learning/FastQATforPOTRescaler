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
from .Type import Data_wrapper


class Start(nn.Module):
    """
    Start Transforms passed values into the quantized/fake quantized domain

    **IMPORTANT** A value domain of [-0.5,0.5] is assumed, fix this of different or force it to that domain

    :param bits: Quantization bit width
    :type bits: int
    """

    @logger_init
    def __init__(self, size=(1,), bits: int = 8) -> None:
        """
        Please read Class help
        """
        super(Start, self).__init__()
        self.size = size
        self.register_buffer("run", (-bits) * torch.ones(size, dtype=torch.float))
        self.register_buffer("delta_in", torch.clone((1.0 / (2.0 ** (-self.run) - 1))).detach())
        self.register_buffer("delta_out", torch.clone((1.0 / (2.0 ** (-self.run)))).detach())
        self.register_buffer("max", 2 ** (-self.run - 1) - 1)
        self.register_buffer("min", -(2 ** (-self.run - 1)))

    @logger_forward
    def forward(self, x: Tensor) -> Data_wrapper:
        return Data_wrapper(
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
    def forward(self, invals: Data_wrapper) -> Tensor:
        x, rexp = invals.get()
        self.exp = rexp.detach().clone()
        with torch.no_grad():
            if not self.training:
                shape = [1 for _ in range(len(x.shape))]
                shape[1] = -1
                x = x.type(self.for_dtype.dtype).mul_(rexp.exp2().view(*shape))
        return x
