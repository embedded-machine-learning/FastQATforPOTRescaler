import torch
import torch.nn as nn
import torch.nn.functional as F

from model_old.layer import *
from model_old.yolo import *

class CAMSPlit(nn.Module):
    def __init__(self):
        super(CAMSPlit, self).__init__()

        self.layers = nn.Sequential(
            Start(-8),
            SplitConvBlockQuant2(3, 16, 3, 2),
            SplitConvBlockQuant2(16, 32, 3, 2),
            SplitConvBlockQuant2(32, 64, 3, 2),
            SplitConvBlockQuant2(64, 64, 3, 2),
            BlockQuant2(64, 64, 3, 1),
            SplitConvBlockQuant2(64, 64, 3, 1),
            SplitConvBlockQuant2(64, 64, 3, 1),
            Conv2dQuant(64, 36, 1, 1),
            Stop(),
        )

        self.yololayer = YOLOLayer(
            [[20, 20], [20, 20], [20, 20], [20, 20], [20, 20], [20, 20]])
        self.yolo_layers = [self.yololayer]

        self.one = torch.ones(1)

    def setquant(self, val):
        # just here for compatability
        pass

    def set_Quant_IG_train_FLAG(self, val):
        # just here for compatability
        pass

    def forward(self, x):
        img_size = x.shape[-2:]
        # print(x.shape)
        yolo_out, out = [], []

        x = self.layers(x)
        # print(torch.max(x.abs()))
        
        # print(x.shape)

        x = self.yololayer(x, img_size)

        yolo_out.append(x)

        if self.training:  # train
            return yolo_out
        else:  # test
            io, p = zip(*yolo_out)  # inference output, training output
            return torch.cat(io, 1), p
        return x

class CAMQuant(nn.Module):
    def __init__(self):
        super(CAMQuant, self).__init__()

        self.layers = nn.Sequential(
            Start(-8),
            BlockQuant2(3, 16, 3, 2),
            BlockQuant2(16, 32, 3, 2),
            BlockQuant2(32, 64, 3, 2),
            BlockQuant2(64, 64, 3, 2),
            BlockQuant2(64, 64, 3, 1),
            BlockQuant2(64, 64, 3, 1),
            BlockQuant2(64, 64, 3, 1),
            Conv2dQuant(64, 36, 1, 1),
            Stop(),
        )

        self.yololayer = YOLOLayer(
            [[20, 20], [20, 20], [20, 20], [20, 20], [20, 20], [20, 20]])
        self.yolo_layers = [self.yololayer]

        self.one = torch.ones(1)

    def setquant(self, val):
        # just here for compatability
        pass

    def set_Quant_IG_train_FLAG(self, val):
        # just here for compatability
        pass

    def forward(self, x):
        img_size = x.shape[-2:]
        # print(x.shape)
        yolo_out, out = [], []

        x = self.layers(x)
        # print(torch.max(x.abs()))
        
        # print(x.shape)

        x = self.yololayer(x, img_size)

        yolo_out.append(x)

        if self.training:  # train
            return yolo_out
        else:  # test
            io, p = zip(*yolo_out)  # inference output, training output
            return torch.cat(io, 1), p
        return x


class CamtadNet_float(nn.Module):
    def __init__(self):
        super(CamtadNet_float, self).__init__()

        self.layers = nn.Sequential(
            Block(3, 16, 3, 2),
            Block(16, 32, 3, 2),
            Block(32, 64, 3, 2),
            Block(64, 64, 3, 2),
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

class CamtadNetSplit_float(nn.Module):
    def __init__(self):
        super(CamtadNetSplit_float, self).__init__()

        self.layers = nn.Sequential(
            SplitBlock(3, 16, 3, 2),
            SplitBlock(16, 32, 3, 2),
            SplitBlock(32, 64, 3, 2),
            SplitBlock(64, 64, 3, 2),
            Block(64, 64, 3, 1),
            SplitBlock(64, 64, 3, 1),
            SplitBlock(64, 64, 3, 1),
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


class CamtadNetFixed(nn.Module):
    def __init__(self):
        super(CamtadNetFixed, self).__init__()

        self.layers = nn.Sequential(
            Start(-8),
            BlockQuant3(3, 16, 3, 2),
            BlockQuant3(16, 32, 3, 2),
            BlockQuant3(32, 64, 3, 2),
            BlockQuant3(64, 64, 3, 2),
            BlockQuant3(64, 64, 3, 1),
            BlockQuant3(64, 64, 3, 1),
            BlockQuant3(64, 64, 3, 1),
            Conv2dQuant(64, 36, 1, 1),
            Stop()
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