import torch
import torch.nn as nn
import torch.nn.functional as F

class CAMSPlit(nn.Module):
    def __init__(self):
        super(CAMSPlit, self).__init__()

        self.layers = nn.Sequential(
            BlockQuant(3, 16, 3, 2),
            BlockQuant(16, 32, 3, 2),
            BlockQuant(32, 64, 3, 2),
            BlockQuant(64, 64, 3, 2),
            BlockQuant(64, 64, 3, 1),
            BlockQuant(64, 64, 3, 1),
            BlockQuant(64, 64, 3, 1),
            BlockQuant(64, 64, 3, 1),
            Conv2dQuant(64, 36, 1, 1),
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

        global running_exp
        running_exp = 0

        x = x * (2**7)
        x = torch.round(x)
        if self.train:
            x = x/(2**7)
        else:
            running_exp = -7

        x = self.layers(x)
        # print(torch.max(x.abs()))
        if not self.training:
            x = x*(2**running_exp)
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