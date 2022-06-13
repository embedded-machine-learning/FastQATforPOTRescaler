
from model.layer import *
from model.convolution import *
from model.yolo import *


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


class CamtadNetFixedPoolN(nn.Module):
    def __init__(self):
        super(CamtadNetFixedPoolN, self).__init__()

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
            Conv2dExpLayerQuantAdaptExp(64, 36, 1, 1),
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

class CamtadNetFixedPoolN2(CamtadNetFixedPoolN):
    def __init__(self):
        super(CamtadNetFixedPoolN2, self).__init__()

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

class CamtadNetFixedPoolN2_highpres_fixed(CamtadNetFixedPoolN):
    def __init__(self):
        super(CamtadNetFixedPoolN2_highpres_fixed, self).__init__()

        self.layers = nn.Sequential(
            Start(8),
            BlockQuantN_fixed(3, 16, 3, 1,outQuantBits=16),
            MaxPool(2,2),
            BlockQuantN_fixed(16, 32, 3, 1,outQuantBits=16),
            MaxPool(2,2),
            BlockQuantN_fixed(32, 64, 3, 1,outQuantBits=16),
            MaxPool(2,2),
            BlockQuantN_fixed(64, 64, 3, 1,outQuantBits=16),
            MaxPool(2,2),
            BlockQuantN_fixed(64, 64, 3, 1,outQuantBits=16),
            BlockQuantN_fixed(64, 64, 3, 1,outQuantBits=16),
            BlockQuantN_fixed(64, 64, 3, 1,outQuantBits=16),
            # Conv2dExpLayerQuantAdaptExp(64, 36, 1, 1),
            BlockQuantNwoA_fixed(64,36,1,1,outQuantBits=16),
            # Bias(36),
            Stop()
        )

class CamtadNetFixedPoolN2_lowpres(CamtadNetFixedPoolN):
    def __init__(self):
        super(CamtadNetFixedPoolN2_lowpres, self).__init__()

        self.layers = nn.Sequential(
            Start(8),
            BlockQuantN_lowpres(3, 16, 3, 1,outQuantBits=8),
            MaxPool(2,2),
            BlockQuantN_lowpres(16, 32, 3, 1,outQuantBits=8),
            MaxPool(2,2),
            BlockQuantN_lowpres(32, 64, 3, 1,outQuantBits=8),
            MaxPool(2,2),
            BlockQuantN_lowpres(64, 64, 3, 1,outQuantBits=8),
            MaxPool(2,2),
            BlockQuantN_lowpres(64, 64, 3, 1,outQuantBits=8),
            BlockQuantN_lowpres(64, 64, 3, 1,outQuantBits=8),
            BlockQuantN_lowpres(64, 64, 3, 1,outQuantBits=8),
            # Conv2dExpLayerQuantAdaptExp(64, 36, 1, 1),
            BlockQuantNwoA_lowpres(64,36,1,1,outQuantBits=16),
            # Bias(36),
            Stop()
        )

class CamtadNetFixedPoolN2_lowpres_2(CamtadNetFixedPoolN):
    def __init__(self):
        super(CamtadNetFixedPoolN2_lowpres_2, self).__init__()

        self.layers = nn.Sequential(
            Start(8),
            BlockQuantN(3, 16, 3, 1,outQuantBits=8),
            MaxPool(2,2),
            BlockQuantN_lowpres(16, 32, 3, 1,outQuantBits=8),
            MaxPool(2,2),
            BlockQuantN_lowpres(32, 64, 3, 1,outQuantBits=4),
            MaxPool(2,2),
            BlockQuantN_lowpres(64, 64, 3, 1,outQuantBits=4),
            MaxPool(2,2),
            BlockQuantN_lowpres(64, 64, 3, 1,outQuantBits=4),
            BlockQuantN_lowpres(64, 64, 3, 1,outQuantBits=8),
            BlockQuantN(64, 64, 3, 1,outQuantBits=8),
            # Conv2dExpLayerQuantAdaptExp(64, 36, 1, 1),
            BlockQuantNwoA(64,36,1,1,outQuantBits=16),
            # Bias(36),
            Stop()
        )

class CamtadNetFixedPoolN2_DynOut(CamtadNetFixedPoolN):
    def __init__(self):
        super(CamtadNetFixedPoolN2_DynOut, self).__init__()

        self.layers = nn.Sequential(
            Start(8),
            BlockQuantN(3, 16, 3, 1,outQuantDyn=True),
            MaxPool(2,2),
            BlockQuantN(16, 32, 3, 1,outQuantDyn=True),
            MaxPool(2,2),
            BlockQuantN(32, 64, 3, 1,outQuantDyn=True),
            MaxPool(2,2),
            BlockQuantN(64, 64, 3, 1,outQuantDyn=True),
            MaxPool(2,2),
            BlockQuantN(64, 64, 3, 1,outQuantDyn=True),
            BlockQuantN(64, 64, 3, 1,outQuantDyn=True),
            BlockQuantN(64, 64, 3, 1,outQuantDyn=True),
            # Conv2dExpLayerQuantAdaptExp(64, 36, 1, 1),
            BlockQuantNwoA(64,36,1,1,outQuantBits=16,outQuantDyn=False),
            # Bias(36),
            Stop()
        )

class CamtadNetFixedPoolN2_fixed(CamtadNetFixedPoolN2):
    def __init__(self):
        super(CamtadNetFixedPoolN2_fixed,self).__init__()
        
        self.layers = nn.Sequential(
            Start(8),
            BlockQuantN_fixed(3, 16, 3, 1),
            MaxPool(2,2),
            BlockQuantN_fixed(16, 32, 3, 1),
            MaxPool(2,2),
            BlockQuantN_fixed(32, 64, 3, 1),
            MaxPool(2,2),
            BlockQuantN_fixed(64, 64, 3, 1),
            MaxPool(2,2),
            BlockQuantN_fixed(64, 64, 3, 1),
            BlockQuantN_fixed(64, 64, 3, 1),
            BlockQuantN_fixed(64, 64, 3, 1),
            # Conv2dExpLayerQuantAdaptExp(64, 36, 1, 1),
            BlockQuantNwoA_fixed(64,36,1,1,outQuantBits=16),
            # Bias(36),
            Stop()
        )

class CamtadNetFixedPoolN2_split(CamtadNetFixedPoolN):
    def __init__(self):
        super(CamtadNetFixedPoolN2_split, self).__init__()

        self.layers = nn.Sequential(
            Start(8),
            BlockQuantN(3, 16, 3, 1),
            MaxPool(2,2),
            BlockQuantN(16, 16, 3, 1,groups=16),
            BlockQuantN(16, 32, 1, 1),
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

class CamtadNetFixedPoolN2_split2(CamtadNetFixedPoolN):
    def __init__(self):
        super(CamtadNetFixedPoolN2_split2, self).__init__()

        self.layers = nn.Sequential(
            Start(8),
            BlockQuantN(3, 16, 3, 1),
            MaxPool(2,2),
            BlockQuantNwoA(16, 16, 3, 1,groups=16),
            BlockQuantN(16, 32, 1, 1),
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

class CamtadNetFixedPoolN2_split2_fixed(CamtadNetFixedPoolN):
    def __init__(self):
        super(CamtadNetFixedPoolN2_split2_fixed, self).__init__()

        self.layers = nn.Sequential(
            Start(8),
            BlockQuantN_fixed(3, 16, 3, 1),
            MaxPool(2,2),
            BlockQuantNwoA_fixed(16, 16, 3, 1,groups=16),
            BlockQuantN_fixed(16, 32, 1, 1),
            MaxPool(2,2),
            BlockQuantN_fixed(32, 64, 3, 1),
            MaxPool(2,2),
            BlockQuantN_fixed(64, 64, 3, 1),
            MaxPool(2,2),
            BlockQuantN_fixed(64, 64, 3, 1),
            BlockQuantN_fixed(64, 64, 3, 1),
            BlockQuantN_fixed(64, 64, 3, 1),
            # Conv2dExpLayerQuantAdaptExp(64, 36, 1, 1),
            BlockQuantNwoA_fixed(64,36,1,1,outQuantBits=16),
            # Bias(36),
            Stop()
        )


# class CamtadNetFloat_marco(CamtadNetFixedPoolN):
#     def __init__(self):
#         super(CamtadNetFloat_marco, self).__init__()

#         self.layers = nn.Sequential(
#             nn.Sequential(),
#             BlockRelu(3, 16, 3, 1),
#             nn.MaxPool2d(2,2),
#             BlockRelu(16, 32, 3, 1),
#             nn.MaxPool2d(2,2),
#             BlockRelu(32, 64, 3, 1),
#             nn.MaxPool2d(2,2),
#             BlockRelu(64, 64, 3, 1),
#             nn.MaxPool2d(2,2),
#             BlockRelu(64, 64, 3, 1),
#             BlockRelu(64, 64, 3, 1),
#             BlockRelu(64, 64, 3, 1),
#             # Conv2dExpLayerQuantAdaptExp(64, 36, 1, 1),
#             BlockReluoA(64,36,1,1),
#             # Bias(36),
#             nn.Sequential(),
#         )

#         self.yololayer = YOLOLayer(
#             [[1.25, 1.25], [2.5, 2.5], [5,5], [10, 10], [20, 20], [20, 20]])
#         self.yolo_layers = [self.yololayer]


class CamtadNetFixedPoolN2_extended(CamtadNetFixedPoolN):
    def __init__(self):
        super(CamtadNetFixedPoolN2_extended, self).__init__()

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
            BlockQuantN(64, 64, 3, 1),  # this one is extra
            # Conv2dExpLayerQuantAdaptExp(64, 36, 1, 1),
            BlockQuantNwoA(64,36,1,1,outQuantBits=16),
            # Bias(36),
            Stop()
        )

class CamtadNetFixed(nn.Module):
    def __init__(self):
        super(CamtadNetFixed, self).__init__()

        self.layers = nn.Sequential(
            Start(8),
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

class CamtadNetFixedBiasChange(nn.Module):
    def __init__(self):
        super(CamtadNetFixedBiasChange, self).__init__()

        self.layers = nn.Sequential(
            Start(8),
            BlockQuantBiasChange(3, 16, 3, 1),
            MaxPool(2,2),
            BlockQuantBiasChange(16, 32, 3, 1),
            MaxPool(2,2),
            BlockQuantBiasChange(32, 64, 3, 1),
            MaxPool(2,2),
            BlockQuantBiasChange(64, 64, 3, 1),
            MaxPool(2,2),
            BlockQuantBiasChange(64, 64, 3, 1),
            BlockQuantBiasChange(64, 64, 3, 1),
            BlockQuantBiasChange(64, 64, 3, 1),
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

class CamtadNetFixedPool(nn.Module):
    def __init__(self):
        super(CamtadNetFixedPool, self).__init__()

        self.layers = nn.Sequential(
            Start(8),
            BlockQuant3(3, 16, 3, 1),
            MaxPool(2,2),
            BlockQuant3(16, 32, 3, 1),
            MaxPool(2,2),
            BlockQuant3(32, 64, 3, 1),
            MaxPool(2,2),
            BlockQuant3(64, 64, 3, 1),
            MaxPool(2,2),
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

class CamtadNetFixedPool2(nn.Module):
    def __init__(self):
        super(CamtadNetFixedPool2, self).__init__()

        self.layers = nn.Sequential(
            Start(8),
            BlockQuant4(3, 16, 3, 1),
            MaxPool(2,2),
            BlockQuant4(16, 32, 3, 1),
            MaxPool(2,2),
            BlockQuant4(32, 64, 3, 1),
            MaxPool(2,2),
            BlockQuant4(64, 64, 3, 1),
            MaxPool(2,2),
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

class CamtadNetFixedPool3(nn.Module):
    def __init__(self):
        super(CamtadNetFixedPool3, self).__init__()

        self.layers = nn.Sequential(
            Start(8),
            BlockQuant4(3, 16, 3, 1),
            MaxPool(2,2),
            BlockQuant4(16, 32, 3, 1),
            MaxPool(2,2),
            BlockQuant4(32, 64, 3, 1),
            MaxPool(2,2),
            BlockQuant4(64, 64, 3, 1),
            MaxPool(2,2),
            BlockQuant4(64, 64, 3, 1),
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


class CamtadNetPool3(nn.Module):
    def __init__(self):
        super(CamtadNetPool3, self).__init__()

        self.layers = nn.Sequential(
            Start(8),
            BlockQuantDyn4(3, 16, 3, 1),
            MaxPool(2,2),
            BlockQuantDyn4(16, 32, 3, 1),
            MaxPool(2,2),
            BlockQuantDyn4(32, 64, 3, 1),
            MaxPool(2,2),
            BlockQuantDyn4(64, 64, 3, 1),
            MaxPool(2,2),
            BlockQuantDyn4(64, 64, 3, 1),
            BlockQuantDyn4(64, 64, 3, 1),
            BlockQuantDyn4(64, 64, 3, 1),
            BlockQuantDyn4(64, 64, 3, 1),
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
            Start(8),
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