import torch.nn as nn
import torch
import numpy as np

from .layer import BlockQuantN,BlockQuantNwoA,Start,Stop
from .yolo import YOLOLayer


class Block(nn.Module):
    def __init__(self, layers_in, layers_out, kernel_size, stride, groups=1) -> None:
        super(Block, self).__init__()

        self.conv = nn.Conv2d(layers_in, layers_out, kernel_size, stride, padding=int(
            np.floor(kernel_size/2)), groups=groups)
        self.bn = nn.BatchNorm2d(layers_out)
        self.prelu = nn.ReLU(0.25)

    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)

        return x

class SplitBlock(nn.Module):
    def __init__(self, layers_in, layers_out, kernel_size, stride, groups=1) -> None:
        super(SplitBlock, self).__init__()

        self.block1 = Block(layers_in,layers_in,kernel_size,stride,layers_in)
        self.block2 = Block(layers_in,layers_out,1,1,1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)

        return x


class CamtadNet_float(nn.Module):
    def __init__(self):
        super(CamtadNet_float, self).__init__()

        self.layers = nn.Sequential(
            Block(3, 16, 3, 1),
            nn.MaxPool2d(2,2),
            Block(16, 32, 3, 1),
            nn.MaxPool2d(2,2),
            Block(32, 64, 3, 1),
            nn.MaxPool2d(2,2),
            Block(64, 64, 3, 1),
            nn.MaxPool2d(2,2),
            Block(64, 64, 3, 1),
            Block(64, 64, 3, 1),
            Block(64, 64, 3, 1),
            nn.Conv2d(64, 36, 1, 1),
        )

        self.yololayer = YOLOLayer(
            [[20, 20], [20, 20], [20, 20], [20, 20], [20, 20], [20, 20]])
        self.yolo_layers = [self.yololayer]


    def set(self, val):
        # just here for compatability
        pass

    def set_Quant_IG_train_FLAG(self, val):
        # just here for compatability
        pass

    def setquant(self, val):
        # just here for compatability
        pass

    def forward(self, x):
        img_size = x.shape[-2:]
        # print(x.shape)
        yolo_out, out = [], []

        x = self.layers(x)
        x = self.yololayer(x, img_size)

        yolo_out.append(x)

        if self.training:  # train
            return yolo_out
        else:  # test
            io, p = zip(*yolo_out)  # inference output, training output
            return torch.cat(io, 1), p
        return x


class CamtadNetQuant(nn.Module):
    def __init__(self):
        super(CamtadNetQuant, self).__init__()

        self.layers = nn.Sequential(
            Start(8),
            BlockQuantN(3, 16, 3, 1),
            MaxPool(2,2),
            BlockQuantN(16, 32, 3, 1),
            MaxPool(2,2),
            BlockQuantN(32, 64, 3, 1),
            MaxPool(2,2),
            BlockQuantN(64, 64, 3, 1),
            MaxPool(2,2),
            BlockQuantN(64, 64, 3, 1),
            BlockQuantN(64, 64, 3, 1),
            BlockQuantN(64, 64, 3, 1),
            # Conv2dExpLayerQuantAdaptExp(64, 36, 1, 1),
            BlockQuantNwoA(64,36,1,1,outQuantBits=16),
            # Bias(36),
            Stop()
        )

        self.yololayer = YOLOLayer(
            [[20, 20], [20, 20], [20, 20], [20, 20], [20, 20], [20, 20]])
        self.yolo_layers = [self.yololayer]

    def convert(self):
        new_layers = nn.Sequential()
        for i in self.layers:
            print(i)
            new_layers.append(i.convert())

        self.layers = new_layers

    def set(self, val):
        # just here for compatability
        pass

    def set_Quant_IG_train_FLAG(self, val):
        # just here for compatability
        pass

    def setquant(self, val):
        # just here for compatability
        pass

    def forward(self, x):
        img_size = x.shape[-2:]
        # print(x.shape)
        yolo_out, out = [], []
        
        x = x-0.5
        x = x.clamp(-0.5,0.5)

        x = self.layers(x)
        x = self.yololayer(x, img_size)

        yolo_out.append(x)

        if self.training:  # train
            return yolo_out
        else:  # test
            io, p = zip(*yolo_out)  # inference output, training output
            return torch.cat(io, 1), p
        return x
