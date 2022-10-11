from .convolution   import Conv2d
from .batchnorm     import BatchNorm2d
from .activations   import LeakReLU
from .layer         import MaxPool2d,AddQAT

import torch.nn as nn
from torch.nn.common_types import _size_any_t,_ratio_any_t,Optional
import torch
import warnings

# from yolov5 repo
def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class ConvQAT(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False,weight_quant_bits=8,weight_quant_channel_wise=True)
        self.bn = BatchNorm2d(c2,out_quant_channel_wise=True,out_quant_bits=8)
        self.act = LeakReLU(2**-4) if act else nn.Sequential()

    def forward(self, x):
        fact = self.bn.get_weight_factor()
        # print("ConvQAT")
        # print(x[0].shape)
        x = self.conv(x, fact)
        # print(x[0].shape)
        x = self.bn(x)
        # print(x[0].shape)
        x = self.act(x)
        # print(x[0].shape)
        return x

    def forward_fuse(self, x):
        return self.forward(x) 
            

class BottleneckQAT(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvQAT(c1, c_, 1, 1)
        self.cv2 = ConvQAT(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
        self.AddQAT = AddQAT(size=(1,c1,1,1),out_quant_bits=8,out_quant_channel_wise=True)

    def forward(self, x):
        # print("BottleneckQAT")
        # print(x[0].shape)
        o1 = self.cv2(self.cv1(x))
        # print("BottleneckQAT continue")
        # print(x[0].shape)
        # print(o1[0].shape)
        return self.AddQAT(x , o1) if self.add else o1


class ConcatQAT(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.dim = dimension

    def forward(self, x):
        # print("ConcatQAT")
        # for q,e in x:
        #     print("q",q.shape)
        #     print("e",e.shape,len(e.shape),len(e.shape)>1,(e.view(1).expand(q.shape[1]).view(1,-1,1,1)).shape)
        vals = torch.cat(tuple([q for q,_ in x]),self.dim)
        rexp = torch.cat(tuple([e if len(e.shape)>1 else e.view(1).expand(q.shape[1]).view(1,-1,1,1) for q,e in x]),self.dim)
        # print(vals.shape)
        # print(rexp.shape)
        return vals,rexp

class C3QAT(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvQAT(c1, c_, 1, 1)
        self.cv2 = ConvQAT(c1, c_, 1, 1)
        self.cv3 = ConvQAT(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(BottleneckQAT(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        self.cat = ConcatQAT(1)

    def forward(self, x):
        return self.cv3(self.cat((self.m(self.cv1(x)), self.cv2(x))))

class SPPFQAT(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = ConvQAT(c1, c_, 1, 1)
        self.cv2 = ConvQAT(c_ * 4, c2, 1, 1)
        self.m = MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cat = ConcatQAT(1)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(self.cat((x, y1, y2, self.m(y2))))



class UpsampleQAT(nn.Upsample):
    def __init__(self, size: Optional[_size_any_t] = None, scale_factor: Optional[_ratio_any_t] = None, mode: str = 'nearest', align_corners: Optional[bool] = None, recompute_scale_factor: Optional[bool] = None) -> None:
        super().__init__(size, scale_factor, mode, align_corners, recompute_scale_factor)

    def forward(self,x):
        if self.training:
            return super(UpsampleQAT,self).forward(x[0]), x[1]
        else:
            return super(UpsampleQAT,self).forward(x[0]), x[1]