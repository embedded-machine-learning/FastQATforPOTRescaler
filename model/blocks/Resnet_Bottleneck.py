from typing import Optional, Callable

import torch.nn as nn


from ..convolution import Conv2d
from ..activations import PACT
from ..batchnorm import BatchNorm2d

from ..DataWrapper import DataWrapper
from ..Quantizer import Quant
from ..add import Add


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> Conv2d:
    """1x1 convolution"""
    return Conv2d(in_planes,
                  out_planes,
                  kernel_size=1,
                  stride=stride,
                  bias=False,
                  weight_quant_bits=8,
                  weight_quant_channel_wise=True,
                  weight_quant=None,
                  )


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> Conv2d:
    """3x3 convolution with padding"""
    return Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
        weight_quant_bits=8,
        weight_quant_channel_wise=True,
        weight_quant=None,
    )


def act(in_planes: int, kargs={}) -> Quant:
    return PACT(8, (1, in_planes, 1, 1), **kargs)


def add(in_planes: int, kargs={}) -> nn.Module:
    return Add((1, in_planes, 1, 1), **kargs)


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.act1 = act(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.act2 = act(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion,out_quant_kargs={"use_enforced_quant_level": True})
        self.act3 = act(width,{"use_enforced_quant_level": True})
        self.downsample = downsample
        self.add = add(planes * self.expansion)
        self.stride = stride

    def forward(self, x: DataWrapper) -> DataWrapper:
        x.set_quant()
        identity = x.clone()

        fact1 = self.bn1.get_weight_factor()
        out = self.conv1(x, fact1)
        out = self.bn1(out, self.act1)


        fact2 = self.bn2.get_weight_factor()
        out = self.conv2(out, fact2)
        out = self.bn2(out, self.act2)


        fact3 = self.bn3.get_weight_factor()
        out = self.conv3(out, fact3)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = self.add(out,identity,self.act3)

        return out
