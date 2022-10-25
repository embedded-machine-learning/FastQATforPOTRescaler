from typing import Optional, Callable

from torch import nn

from ..Quantizer import Quant
from ..DataWrapper import DataWrapper
from ..logger import logger_forward,logger_init
from ..batchnorm import BatchNorm2d
from ..convolution import Conv2d
from ..activations import PACT
from ..add import Add


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
        # weight_quant_bits=8,
        # weight_quant_channel_wise=True,
        weight_quant=None,
    )

def act(in_planes:int,kargs={}) -> Quant:
    return PACT(8,(1,in_planes,1,1),**kargs)

def add(in_planes:int,kargs={})-> nn.Module:
    return Add((1,in_planes,1,1),**kargs)

class ResidualBlock(nn.Module):
    """
    ResidualBlock The Residual Block from Resnet

    :param inplanes: Number of input planes
    :type inplanes: int
    :param planes: Number of output planes
    :type planes: int
    :param stride: The Stride, defaults to 1
    :type stride: int, optional
    :param downsample: The down-sample Module, defaults to None
    :type downsample: Optional[nn.Module], optional
    :param groups: The size of the Groups, defaults to 1
    :type groups: int, optional
    :param base_width: Jeah no Idea go look at resnet , defaults to 64
    :type base_width: int, optional
    :param dilation: The dilation, defaults to 1
    :type dilation: int, optional
    :param norm_layer: The Normlaer used, if None the BatchNorm2d is used (you know the QAT one), defaults to None
    :type norm_layer: Optional[Callable[..., nn.Module]], optional
    """
    expansion: int = 1
    @logger_init
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
        super(ResidualBlock,self).__init__()
        if norm_layer is None:
            norm_layer = BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.act1 = act(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.act2 = act(planes,{'use_enforced_quant_level':True})
        self.downsample = downsample
        self.stride = stride
        self.add = add(planes)
        self.act3 = act(planes,{'use_enforced_quant_level':True})

    @logger_forward
    def forward(self, x: DataWrapper) -> DataWrapper:
        x.set_quant()

        fact1 = self.bn1.get_weight_factor()
        out = self.conv1(x, fact1)
        out = self.bn1(out,self.act1)

        fact2 = self.bn2.get_weight_factor()
        out = self.conv2(out, fact2)
        out = self.bn2(out,self.act2)

        if self.downsample is not None:
            x = self.downsample(x)

        out = self.add(out, x, self.act3)

        return out