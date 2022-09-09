import warnings
from functools import partial
from typing import Callable, Any, Optional, List, Tuple

import torch
from torch import Tensor
from torch import nn
import torchvision

from torchvision.models._utils import _make_divisible

from .batchnorm import BatchNorm2d
from .layer import AdaptiveAvgPool2d, AddQAT, Flatten, Start, Stop
from .Linear import Linear
from .convolution import Conv2d
from .quantizer import F8NetQuant
from .activations import PACT_fused_F8NET_mod, ReLU_F8NET_fused

from .Linear import LinQuantWeight_mod_F8NET as Lin_Weight_quant_mod_F8NET
from .convolution import LinQuantWeight_mod_F8NET as Conv_Weight_quant_mod_F8NET


__all__ = ["MobileNetV2", "MobileNet_V2_Weights", "mobilenet_v2"]


def Conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: Optional[int] = None,
    groups: int = 1,
    dilation: int = 1,
):
    return Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=False,
        weight_quant=Conv_Weight_quant_mod_F8NET(bits=8, size=(out_channels, 1, 1, 1),layer_wise=(out_channels==groups)),
    )


def Lin(in_features: int, out_features: int, bias: bool = True):
    return Linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        weight_quant=Lin_Weight_quant_mod_F8NET(bits=8, size=(out_features, 1)),
        out_quant=F8NetQuant(bits=16, size=(1, out_features)),
        out_quant_channel_wise=True
    )


def PACT_F8NET(features):
    return PACT_fused_F8NET_mod(bits=8, size=(1, 1, 1, 1))
    # return PACT_fused_F8NET_mod(bits=8, size=(1, features, 1, 1))


def F8NET_Quant(features):
    return F8NetQuant(bits=8, size=(1, 1, 1, 1))
    # return F8NetQuant(bits=8, size=(1, features, 1, 1))


def default_Quant(features):
    return F8NET_Quant(features)


def default_fused_Activation(features):
    return PACT_F8NET(features)


def ADDwQuant(planes: int):
    return AddQAT(
        size=(1, planes, 1, 1),
        out_quant=default_Quant(planes),
    )


class Conv2dNormActivation(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        dilation: int = 1,
    ) -> None:
        super(Conv2dNormActivation,self).__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation

        self.conv = Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
        )
        self.bn = BatchNorm2d(num_features=out_channels, out_quant=default_fused_Activation(out_channels))

    def forward(self, x: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        fact = self.bn.get_weight_factor()
        x = self.conv(x, fact)
        x = self.bn(x)
        return x

class Conv2dNorm(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        dilation: int = 1,
    ) -> None:
        super(Conv2dNorm,self).__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation

        self.conv = Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
        )
        self.bn = BatchNorm2d(num_features=out_channels, out_quant=default_Quant(out_channels))

    def forward(self, x: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        fact = self.bn.get_weight_factor()
        x = self.conv(x, fact)
        x = self.bn(x)
        return x


class InvertedResidual(nn.Module):
    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: int) -> None:
        super().__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 insted of {stride}")

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        if self.use_res_connect:
            self.add = ADDwQuant(oup)

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(
                Conv2dNormActivation(inp, hidden_dim, kernel_size=1)
            )
        layers.extend(
            [
                # dw
                Conv2dNormActivation(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                ),
                # pw-linear
                Conv2dNorm(hidden_dim,oup,1,1,0)
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self,  x: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        if self.use_res_connect:
            return self.add(x,self.conv(x))
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.2,
    ) -> None:
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
            dropout (float): The droupout probability

        """
        super().__init__()

        if block is None:
            block = InvertedResidual

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError(
                f"inverted_residual_setting should be non-empty or a 4-element list, got {inverted_residual_setting}"
            )

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features: List[nn.Module] = [
            Conv2dNormActivation(
                3,
                input_channel,
                stride=2,
            )
        ]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(
            Conv2dNormActivation(
                input_channel,
                self.last_channel,
                kernel_size=1,
            )
        )
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            # nn.Dropout(p=dropout),                    # no Dropout in QAT
            Lin(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

        self.start = Start(8)
        self.stop = Stop(size=(1, num_classes))
        self.avgpool = AdaptiveAvgPool2d((1, 1))

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = x - 0.5  # make a symetric input
        x = self.start(x)  # convert to QAT domain

        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1
        x = self.avgpool(x)
        x = Flatten(x, 1)
        x = self.classifier(x)

        x = self.stop(x)  # recover from QAT domain
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)