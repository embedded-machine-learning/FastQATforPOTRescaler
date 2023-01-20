import torchvision
from typing import Tuple, Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn


from torchvision._internally_replaced_utils import load_state_dict_from_url
from torchvision.utils import _log_api_usage_once


from ..blocks import ResidualBlock as BasicBlock
from ..blocks import Bottleneck
from ..blocks import ConvBnA, ConvBn
from ..wrapped import MaxPool2d, AdaptiveAvgPool2d, Flatten
from ..batchnorm import BatchNorm2d
from ..linear import Linear
from ..Conversion import Start, Stop


from ..convolution import Conv2d
from ..activations import PACT

####################################################################################
# This is mostly copied and modivied from torchvision/jmodels/resnet.py
####################################################################################


def Downsample_Block(inplanes, planes, expansion, stride):
    return ConvBn(
        in_channels=inplanes,
        out_channels=planes * expansion,
        kernel_size=1,
        stride=stride,
        out_quant_kargs={"use_enforced_quant_level": True},
    )

def LinearHelper(in_features,out_features):
    return Linear(in_features=in_features,
        out_features=out_features,
        bias=True,
        weight_quant=None,
        weight_quant_bits=8,
        weight_quant_channel_wise=True,
        )


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False,weight_quant_channel_wise=True)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = PACT(8,(1,self.inplanes,1,1))

        self.maxpool = MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        self.avgpool = AdaptiveAvgPool2d((1, 1))
        self.fc = LinearHelper(512 * block.expansion, num_classes)

        self.start = Start(8)
        self.stop = Stop((1, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    # type: ignore[arg-type]
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    # type: ignore[arg-type]
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Downsample_Block(
                self.inplanes, planes, block.expansion, stride)

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # See note [TorchScript super()]
        x = x - 0.5  # make symetric input
        x = self.start(x)

        fact1 = self.bn1.get_weight_factor()
        x = self.conv1(x, fact1)
        x = self.bn1(x,self.relu)

        # relu is fused into BN
        # x = checkNanTuple(x,"forw. pre max pool")
        x = self.maxpool(x)
        # x = checkNanTuple(x,"forw. post max pool")
        x = self.layer1(x)
        # x = checkNanTuple(x,"forw. post layer 1")
        x = self.layer2(x)
        # x = checkNanTuple(x,"forw. post layer 2")
        x = self.layer3(x)
        # x = checkNanTuple(x,"forw. post layer 3")
        x = self.layer4(x)
        # x = checkNanTuple(x,"forw. post layer 4")

        x = self.avgpool(x)
        x = Flatten(x, 1)
        x = self.fc(x)

        x = self.stop(x)
        # x = checkNan.apply(x,"After stop")

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(torchvision.models.resnet.model_urls[arch], progress=progress)
        model.load_state_dict(state_dict,strict=False)
    return model



def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)