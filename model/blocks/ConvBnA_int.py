from typing import Union

import torch
from torch import nn
from torch.nn.common_types import Tensor, _size_2_t

from ..logger import logger_forward, logger_init
from .. import __FLAGS__


class ConvBnA_int(nn.Module):
    """
    ConvBnA_int A integer converted ConvBnA block

    Is not meant for training only for validation of the trained results

    :param Conv_weight: _description_
    :type Conv_weight: _type_
    :param BN_shift: _description_
    :type BN_shift: _type_
    :param BN_add: _description_
    :type BN_add: _type_
    :param Act_min: _description_
    :type Act_min: _type_
    :param Act_max: _description_
    :type Act_max: _type_
    """

    @logger_init
    def __init__(
        self,
        # Convolution definition
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t,
        padding: Union[str, _size_2_t],
        dilation: _size_2_t,
        groups: int,
        # transferred Convolution parameter
        Conv_weight: Tensor,
        BN_shift: Tensor,
        BN_add: Tensor,
        Act_min: Tensor,
        Act_max: Tensor,
        # type information
        accumulation_type=torch.int32,
        small_signed_type=torch.int8,
        small_unsigned_type=torch.uint8,
    ) -> None:
        super(ConvBnA_int, self).__init__()
        self.Conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.Conv.weight.requires_grad_(False)
        self.Conv.weight.data = Conv_weight.to(small_signed_type)

        self.register_buffer("n", BN_shift.int())
        self.register_buffer("n_eq_mult", BN_shift.exp2())

        self.register_buffer("t", BN_add.to(accumulation_type))
        self.register_buffer("min", Act_min.to(accumulation_type))
        self.register_buffer("max", Act_max.to(accumulation_type))

        self.pure_positive = torch.all(Act_min>=0).cpu().numpy()

        self.accumulation_type = accumulation_type
        self.small_signed_type = small_signed_type
        self.small_unsigned_type = small_unsigned_type

    @logger_forward
    def forward(self, x: Tensor) -> Tensor:
        if __FLAGS__["ONNX_EXPORT"]:
            x = self.Conv._conv_forward(x.float(), self.Conv.weight.float(), self.Conv.bias).to(self.accumulation_type)
        else:
            x = self.Conv._conv_forward(x.to(self.accumulation_type), self.Conv.weight.to(self.accumulation_type), self.Conv.bias)

        x = x + self.t

        if __FLAGS__["ONNX_EXPORT"]:
            x = x.to(torch.float).mul(self.n_eq_mult).floor().to(self.n.dtype)
        else:
            x = torch.bitwise_right_shift(x, -self.n)

        x = x.clamp(self.min, self.max)
        if self.pure_positive:
            x = x.to(self.small_unsigned_type)
        else:
            x = x.to(self.small_signed_type)
        return x
