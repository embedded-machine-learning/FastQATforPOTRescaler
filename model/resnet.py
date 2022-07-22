from typing import Tuple, Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn

from torchvision.utils          import _log_api_usage_once
from torchvision.models._api    import WeightsEnum
from torchvision.models._utils  import _ovewrite_named_param

from .convolution   import Conv2dQuant_new
from .batchnorm     import BatchNorm2dBase_new
from .activations   import ReLU
from .layer         import AddQAT, MaxPool2d, Start, Stop, AdaptiveAvgPool2d, Flatten
from .Linear        import Linear

####################################################################################
# This is mostly copied and modivied from torchvision/jmodels/resnet.py
####################################################################################



def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> Conv2dQuant_new:
    """3x3 convolution with padding"""
    return Conv2dQuant_new(
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
        out_quant_bits=8,
        out_quant_channel_wise=True
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> Conv2dQuant_new:
    """1x1 convolution"""
    return Conv2dQuant_new(
        in_planes, 
        out_planes, 
        kernel_size=1, 
        stride=stride, 
        bias=False,
        weight_quant_bits=8,
        weight_quant_channel_wise=True,
        out_quant_bits=8,
        out_quant_channel_wise=True
    )

class Downsample_Block(nn.Module):
    def __init__(self,inplanes,planes,expansion,stride) -> None:
        super().__init__()
        self.inplanes   = inplanes
        self.planes     = planes
        self.expansion  = expansion
        self.stride     = stride

        self.conv   = conv1x1(inplanes,planes*expansion,stride)
        self.bn     = BatchNorm2dBase_new(planes * expansion)

    def forward(self,x):
        fact1 = self.bn.get_weight_factor()
        x = self.conv(x,fact1)
        x = self.bn(x,self.conv.quantw.delta.detach())
        return x


class BasicBlock(nn.Module):
    expansion: int = 1

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
            norm_layer = BatchNorm2dBase_new
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.add = AddQAT()

    def forward(self, x: Tuple[torch.Tensor,torch.Tensor]) -> Tuple[torch.Tensor,torch.Tensor]:
        identity = x
        fact1 = self.bn1.get_weight_factor()
        out = self.conv1(x,fact1)
        out = self.bn1(out,self.conv1.quantw.delta.detach())
        out = self.relu(out)
        fact2 = self.bn2.get_weight_factor()
        out = self.conv2(out,fact2)
        out = self.bn2(out,self.conv2.quantw.delta.detach())
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.add(out,identity)
        out = self.relu(out)
        return out


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
            norm_layer = BatchNorm2dBase_new
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride
        self.add = AddQAT()

    def forward(self, x: Tuple[torch.Tensor,torch.Tensor]) -> Tuple[torch.Tensor,torch.Tensor]:
        identity = x

        fact1 = self.bn1.get_weight_factor()
        out = self.conv1(x,fact1)
        out = self.bn1(out,self.conv1.quantw.delta.detach())
        out = self.relu(out)


        fact2 = self.bn2.get_weight_factor()
        out = self.conv2(out,fact2)
        out = self.bn2(out,self.conv2.quantw.delta.detach())
        out = self.relu(out)


        fact3 = self.bn3.get_weight_factor()
        out = self.conv3(out,fact3)
        out = self.bn3(out,self.conv3.quantw.delta.detach())

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.add(out,identity)
        out = self.relu(out)

        return out


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
            norm_layer = BatchNorm2dBase_new
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
        self.conv1 = Conv2dQuant_new(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False,weight_quant_bits=8,weight_quant_channel_wise=True)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = ReLU(inplace=False)
        self.maxpool = MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        self.avgpool = AdaptiveAvgPool2d((1, 1))
        self.fc = Linear(512 * block.expansion, num_classes,weight_quant_channel_wise=True,out_quant_channel_wise=True,out_quant_bits=16)

        self.start = Start(8)
        self.stop = Stop()

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
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

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
            downsample = Downsample_Block(self.inplanes,planes,block.expansion,stride) 

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

        x = self.start(x)

        
        fact1 = self.bn1.get_weight_factor()
        x = self.conv1(x,fact1)
        x = self.bn1(x,self.conv1.quantw.delta.detach())
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)



        x = self.avgpool(x)
        x = Flatten(x,1)
        x = self.fc(x)

        x = self.stop(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)


def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model




import torchvision

def resnet18(*, weights = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-18 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.

    Args:
        weights (:class:`~torchvision.models.ResNet18_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet18_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet18_Weights
        :members:
    """
    weights = torchvision.models.resnet.ResNet18_Weights.verify(weights)

    return _resnet(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)
