
from model.layer import *
from model.convolution import *
from model.yolo import *


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
            Conv2dExpLayerQuantNormWeightsAdaptExp(64, 36, 1, 1),
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

class CamtadNetFixed2(nn.Module):
    def __init__(self):
        super(CamtadNetFixed2, self).__init__()

        self.layers = nn.Sequential(
            Start(-8),
            BlockQuant4(3, 16, 3, 2),
            BlockQuant4(16, 32, 3, 2),
            BlockQuant4(32, 64, 3, 2),
            BlockQuant4(64, 64, 3, 2),
            BlockQuant4(64, 64, 3, 1),
            BlockQuant4(64, 64, 3, 1),
            BlockQuant4(64, 64, 3, 1),
            Conv2dExpLayerQuantAdaptExp(64, 36, 1, 1),
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