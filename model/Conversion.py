# Generic Type imports
from types import NoneType
from typing import Tuple, Union

# Torch imports
import torch
from torch import nn
from torch.nn.common_types import Tensor

# Self init
from .logger import logger_init, logger_forward
from .Quantizer import FakeQuant
from .DataWrapper import DataWrapper


class Start_int(nn.Module):
    @logger_init
    def __init__(self, delta_in: Tensor, min: Tensor, max: Tensor) -> None:
        super(Start_int, self).__init__()
        self.register_buffer("delta_in", delta_in)
        self.register_buffer("min", min)
        self.register_buffer("max", max)

    def forward(self, x: Tensor) -> Tensor:
        x = x.div(self.delta_in, rounding_mode="floor")
        x = x.type(torch.int8)
        x = x.clamp(self.min, self.max)
        return x


class Start(nn.Module):
    """
    Start Transforms passed values into the quantized/fake quantized domain

    **IMPORTANT** A value domain of [-0.5,0.5] is assumed, fix this of different or force it to that domain

    :param bits: The number of desired bits, defaults to 8
    :type bits: int, optional
    :param size: The shape of the quantization, defaults to (1,)
    :type size: Tuple[int], optional
    :param mode: Defines if the quantization range should be extracted during runtime or set to [-.5,.5], defaults to "auto"
    :type mode: Union[str, NoneType], optional
    :param auto_runs: Number of epochs to fixate the auto range, defaults to 2
    :type auto_runs: int, optional
    """

    @logger_init
    def __init__(
        self, bits: int = 8, size: Tuple[int] = (1,), mode: Union[str, NoneType] = "auto", auto_runs: int = 2
    ) -> None:
        """
        Please read Class help
        """
        super(Start, self).__init__()
        self.size = size
        self.register_buffer("run", (-bits) * torch.ones(size, dtype=torch.float))
        self.register_buffer("delta_in", torch.clone((1.0 / (2.0 ** (-self.run) - 1))).detach())
        self.register_buffer("delta_out", torch.clone((1.0 / (2.0 ** (-self.run) - 1))).detach())
        self.register_buffer("max", 2 ** (-self.run - 1) - 1)
        self.register_buffer("min", -(2 ** (-self.run - 1)))

        self.mode = mode
        self.auto_runs = auto_runs
        self.last_run_train = True
        self.register_buffer("in_max", torch.Tensor([-10000.0]))
        self.register_buffer("in_min", torch.Tensor([10000.0]))

    def int_extract(self,type_small=torch.int8,type_big=torch.int32) -> Start_int:
        return Start_int(self.delta_in, self.min.type(type_big), self.max.type(type_big))

    @logger_forward
    def forward(self, x: Tensor) -> DataWrapper:
        with torch.no_grad():
            if self.training:
                if self.mode == "auto" and self.auto_runs > 0:
                    self.last_run_train = True
                    self.in_min = torch.min(torch.min(x), self.in_min)
                    self.in_max = torch.max(torch.max(x), self.in_max)
                    rang = 2 * (torch.max(torch.abs(self.in_min), torch.abs(self.in_max)))
                    self.delta_in = rang / (2.0 ** (-self.run) - 1)
                    self.delta_out = rang / (2.0 ** (-self.run) - 1)
            else:
                if self.last_run_train:
                    self.last_run_train = False
                    self.auto_runs -= 1

        return DataWrapper(
            FakeQuant(
                x.clone(),
                self.delta_in,
                self.delta_out,
                self.training,
                self.min,
                self.max,
                "floor",
            ),
            torch.log2(self.delta_out),
        )

class Stop_int(nn.Module):
    def __init__(self,rexp) -> None:
        super(Stop_int,self).__init__()
        self.register_buffer('rexp',rexp)
        self.register_buffer('mult_factor',rexp.exp2())
        
    def forward(self,x:Tensor)->Tensor:
        # print(x)
        # print(self.rexp)
        return x.type(torch.float32).mul(self.mult_factor)

class Stop(nn.Module):
    """
    Stop Return a Tensor pair from the fake-quantized/quantized domain

    :param size: The shape of the output, defaults to (1,)
    :type size: Tuple[int], optional
    """

    @logger_init
    def __init__(self, size: Tuple[int] = (1,)) -> None:
        """
        Please read Class help
        """
        super(Stop, self).__init__()
        self.size = size
        self.register_buffer("exp", torch.zeros(self.size))
        self.register_buffer("for_dtype", torch.zeros(1))  # Only required to know the current datatype

    def int_extract(self)-> Stop_int:
        return Stop_int(self.exp)

    @logger_forward
    def forward(self, invals: DataWrapper) -> Tensor:
        x, rexp = invals.get()
        self.exp = rexp.detach().clone()
        with torch.no_grad():
            if not self.training:
                shape = [1 for _ in range(len(x.shape))]
                shape[1] = -1
                x = x.type(self.for_dtype.dtype).mul_(rexp.exp2().view(*shape))
        return x
