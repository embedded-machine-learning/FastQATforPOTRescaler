from .convolution   import Conv2dQuant_new
from .batchnorm     import BatchNorm2dBase_new
from .activations   import LeakReLU
from .layer         import MaxPool

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
        self.conv = Conv2dQuant_new(c1, c2, k, s, autopad(k, p), groups=g, bias=False,weight_quant_bits=32,weight_quant_channel_wise=True)
        self.bn = BatchNorm2dBase_new(c2,outQuantDyn=True,outQuantBits=32)
        self.act = LeakReLU(0.125) if act else nn.Sequential()

    def forward(self, x):
        fact = self.bn.get_weight_factor().detach()
        # print("ConvQAT")
        # print(x[0].shape)
        x = self.conv(x, fact)
        # print(x[0].shape)
        x = self.bn(x, self.conv.quantw.delta.detach())
        # print(x[0].shape)
        x = self.act(x)
        # print(x[0].shape)
        return x

    def forward_fuse(self, x):
        return self.forward(x) 
        
# def AddQAT(a,b):
#     arexp = a[1]
#     brexp = b[1]
#     return a[0]+b[0],a[1]


class AddQAT_(torch.autograd.Function):
    @staticmethod
    def forward(_,a,b,a_shift,b_shift,rexp,training):
        with torch.no_grad():
            if training:
                va = (a*torch.exp2(-rexp).view(-1)[None,:,None,None]).floor()
                vb = (b*torch.exp2(-rexp).view(-1)[None,:,None,None]).floor()
            else:
                va = a.mul(torch.exp2(-a_shift).view(-1)[None,:,None,None]).floor()
                vb = b.mul(torch.exp2(-b_shift).view(-1)[None,:,None,None]).floor()
            #explicit quant domaine
            va = va.add(vb)

            #done
            if training:
                va = va.mul(torch.exp2(rexp).view(-1)[None,:,None,None])

            return va

    @staticmethod
    def backward(_,outgrad):
        return outgrad.detach(),outgrad.detach(),None,None,None,None

class AddQAT(nn.Module):
    def __init__(self) -> None:
        super(AddQAT,self).__init__()

        self.register_buffer('a_shift',torch.Tensor([0.0]))
        self.register_buffer('b_shift',torch.Tensor([0.0]))

    def forward(self,a,b):
        if a[0].shape!=b[0].shape:
            raise torch.ErrorReport("testW")
        arexp = a[1]
        brexp = b[1]
        rexp = torch.max(arexp,brexp)
        self.a_shift = -(arexp-rexp).detach()
        self.b_shift = -(brexp-rexp).detach()
        out = AddQAT_.apply(a[0],b[0],self.a_shift,self.b_shift,rexp,self.training)
        # print("AddQAT")
        # print(out.shape,a[0].shape)
        return out,rexp


            

class BottleneckQAT(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvQAT(c1, c_, 1, 1)
        self.cv2 = ConvQAT(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
        self.AddQAT = AddQAT()

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
        self.m = MaxPool(kernel_size=k, stride=1, padding=k // 2)
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
        return super(UpsampleQAT,self).forward(x[0]), x[1]