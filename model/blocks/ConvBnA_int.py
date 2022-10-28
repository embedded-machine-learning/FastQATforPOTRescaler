from typing import Union

import torch
from torch import nn
from torch.nn.common_types import Tensor,_size_2_t

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
        Conv_weight:Tensor,
        BN_shift:Tensor,
        BN_add:Tensor,
        Act_min:Tensor,
        Act_max:Tensor,
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
        self.Conv.weight.data=Conv_weight

        self.register_buffer("n",BN_shift)
        self.register_buffer("n_eq_mult",BN_shift.exp2())

        self.register_buffer("t",BN_add)
        self.register_buffer("min",Act_min)
        self.register_buffer("max",Act_max)
    
    @logger_forward
    def forward(self,x:Tensor)-> Tensor:
        x = self.Conv(x)
        x = x+self.t

        if __FLAGS__['ONNX_EXPORT']:
            x = x.type(torch.float).mul(self.n_eq_mult).floor().type(torch.int)
        else:
            x = torch.bitwise_right_shift(x,-self.n)

        x = x.clamp(self.min,self.max)
        return x
