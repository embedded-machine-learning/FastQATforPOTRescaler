import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_any_t, _size_any_opt_t

from typing import Optional

from ..logger import logger_forward, logger_init
from ..DataWrapper import DataWrapper


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
        assert return_indices == False

    def forward(self, input: DataWrapper):
        val, rexp = input.get()
        return input.set(
            F.max_pool2d(
                val, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode, self.return_indices
            ),
            rexp,
        )



class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    def __init__(self, output_size: _size_any_opt_t) -> None:
        super(AdaptiveAvgPool2d,self).__init__(output_size)

    def forward(self, input:DataWrapper) -> DataWrapper:
        # does nopt modify the channels so simple wrapping and floor should be enough
        val, rexp = input.get()

        val = super(AdaptiveAvgPool2d,self).forward(val)

        if self.training:
            with torch.no_grad():
                val.data = val.data.div_(torch.exp2(rexp),rounding_mode="floor")
                val.data = val.data.mul_(torch.exp2(rexp))

        else:
            val = val.floor_()

        return input.set(val, rexp)

