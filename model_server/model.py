import torch
import torch.nn as nn
import torch.nn.functional as F

from model_server.layer import *
from model.yolo import *

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
            Start(-7),
            BlockQuant3(3, 16, 3, 2),
            BlockQuant3(16, 32, 3, 2),
            BlockQuant3(32, 64, 3, 2),
            BlockQuant3(64, 64, 3, 2),
            BlockQuant3(64, 64, 3, 1),
            BlockQuant3(64, 64, 3, 1),
            BlockQuant3(64, 64, 3, 1),
            Conv2dQuant(64, 36, 1, 1),
            Stop((1,36,1,1))
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
        return 
        
class QTCamtadNetFixed(nn.Module):
    def __init__(self):
        super(QTCamtadNetFixed, self).__init__()

        self.n = []
        self.t = []
        self.num_bits = 8
        self.exponent = 0
        self.run = 0

        self.layers = nn.Sequential(
            #Start(-8),
            
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=int(np.floor(3/2)), groups=1, bias = False),
            nn.LeakyReLU(),
            #BlockQuant3(3, 16, 3, 2),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=int(np.floor(3/2)), groups=1, bias = False),
            nn.LeakyReLU(),
            #BlockQuant3(16, 32, 3, 2),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=int(np.floor(3/2)), groups=1, bias = False),
            nn.LeakyReLU(),
            #BlockQuant3(32, 64, 3, 2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=int(np.floor(3/2)), groups=1, bias = False),
            nn.LeakyReLU(),
            #BlockQuant3(64, 64, 3, 2),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=int(np.floor(3/2)), groups=1, bias = False),
            nn.LeakyReLU(),
            #BlockQuant3(64, 64, 3, 1),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=int(np.floor(3/2)), groups=1, bias = False),
            nn.LeakyReLU(),
            #BlockQuant3(64, 64, 3, 1),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=int(np.floor(3/2)), groups=1, bias = False),
            nn.LeakyReLU(),
            #BlockQuant3(64, 64, 3, 1),

            nn.Conv2d(64, 36, kernel_size=1, stride=1, padding=int(np.floor(1/2)), groups=1, bias = False),
            #Conv2dQuant(64, 36, 1, 1),
            #Stop()
        )

        self.yololayer = YOLOLayer(
            [[20, 20], [20, 20], [20, 20], [20, 20], [20, 20], [20, 20]])
        self.yolo_layers = [self.yololayer]

    def forward(self, x):
        img_size = x.shape[-2:]
        # print(x.shape)
        yolo_out, out = [], []

        min_val = torch.tensor(-(1 << (self.num_bits - 1)))
        max_val = torch.tensor((1 << (self.num_bits - 1))-1)

        x = x*(2**(-self.run))
        x = torch.round(x)
        if torch.max(torch.abs(x)) > 128:
            print(torch.max(torch.abs(x)))
        x = torch.clamp(x,min_val,max_val)

        for i, layer in enumerate(self.layers):
            if i < len(self.layers)-1:
                if "Conv2d" in str(layer):
                    x = layer(x)
                    x = torch.round(x)
                    if torch.max(torch.abs(x)) > 2**16:
                        print("conv2d" + str(torch.max(torch.abs(x))))
                    x = x*torch.exp2(self.n[i])[None, :, None, None] + self.t[i][None, :, None, None]
                    x = torch.round(x)
                    # print("conv2d scale" + str(torch.max(torch.abs(x))))
                    x = torch.clamp(x,min_val,max_val)
                elif "LeakyReLU" in str(layer):
                    x = layer(x)
                    x = torch.round(x)
                    # print("relu" + str(torch.max(torch.abs(x)))
                    x = torch.clamp(x,min_val,max_val)
                else:
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!")
            elif "Conv2d" in str(layer):
                x = layer(x)
                x = torch.round(x)
                if torch.max(torch.abs(x)) > 2**16:
                    print("last" + str(torch.max(torch.abs(x))))
                x = x/(2**self.exponent)
                # print("last_float" + str(torch.max(torch.abs(x))))

            else:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!")

        x = self.yololayer(x, img_size)

        yolo_out.append(x)

        if self.training:  # train
            return yolo_out
        else:  # test
            io, p = zip(*yolo_out)  # inference output, training output
            return torch.cat(io, 1), p
        return x