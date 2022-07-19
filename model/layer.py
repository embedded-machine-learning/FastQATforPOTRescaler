import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch.nn.common_types import _size_any_t, Tensor
from .batchnorm import *

import numpy as np

from .utils import *
from .convolution import *
from .activations import *

#########################################################################################
#                                   FUNCTIONS                                           #
#########################################################################################


class Startfn(torch.autograd.Function):
    @staticmethod
    def forward(self, x: Tensor, delta, rexp: Tensor, training: bool) -> Tensor:
        with torch.no_grad():
            x = x.div(delta, rounding_mode="floor")
            x = x.clamp(-2**(-rexp-1),2**(-rexp-1)-1)
            if training:
                x = x.div(2**(-rexp))
            else :
                x = x.type(torch.int32)
        return x

    @staticmethod
    def backward(self, x: Tensor):
        return x.detach(), None, None, None


class Stopfn(torch.autograd.Function):
    @staticmethod
    def forward(self, val: Tensor, rexp: Tensor, training: bool,dtype):
        with torch.no_grad():
            if not training:
                val = val.type(dtype).div(2**(-rexp.view(-1)[None, :, None, None]))
        return val

    @staticmethod
    def backward(self, x: Tensor):
        return x.detach(), None, None, None

#########################################################################################
#                                   CONVERSIONS                                         #
#########################################################################################


class BlockQuantN_(nn.Module):
    def __init__(self, conv, bn, act) -> None:
        super(BlockQuantN_, self).__init__()
        self.conv = conv.convert()
        self.bn = bn.convert()
        if type(act) != nn.Sequential:
            self.activation = act.convert()
        else:
            self.activation = act

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class Start_(nn.Module):
    def __init__(self, run, delta) -> None:
        super(Start_, self).__init__()
        self.register_buffer('run', run.clone())
        self.delta = delta

    def forward(self, x):
        rexp = self.run
        x = x/self.delta
        x = Floor.apply(x)
        return x


class Stop_(nn.Module):
    def __init__(self, rexp: torch.Tensor()) -> None:
        super(Stop_, self).__init__()
        self.register_buffer("rexp", rexp.clone())

    def forward(self, x: torch.Tensor):
        x = x/(2**-self.rexp[None, :, None, None])
        x = checkNan.apply(x,"Stop_")       # removes nan from backprop
        return x

#########################################################################################
#                                   CLASSES                                             #
#########################################################################################


class Start(nn.Module):
    def __init__(self, running_exp_init) -> None:
        super(Start, self).__init__()
        self.register_buffer('run', torch.tensor(
            [-running_exp_init], dtype=torch.float))
        self.register_buffer("delta", torch.tensor([1.0/(2.0**(-self.run)-1)]))

    def convert(self):
        return Start_(self.run, self.delta)

    def forward(self, x: Tensor):
        x = Startfn.apply(x, self.delta, self.run, self.training)
        return x, self.run


class Stop(nn.Module):
    def __init__(self) -> None:
        super(Stop, self).__init__()
        self.size = []
        self.register_buffer('exp', torch.zeros(1))
        self.register_buffer('for_dtype', torch.zeros(1))


    def convert(self):
        return Stop_(self.exp)

    def forward(self, invals: Tuple[Tensor, Tensor]) -> Tensor:
        self.exp = invals[1].detach().clone()
        x = Stopfn.apply(invals[0], invals[1], self.training, self.for_dtype.dtype)
        x = checkNan.apply(x,"Stop")       # removes nan from backprop
        return x


class Bias(nn.Module):
    def __init__(self, num_features, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Bias, self).__init__()
        self.bias = torch.nn.Parameter(
            torch.empty(num_features, **factory_kwargs))
        torch.nn.init.zeros_(self.bias)
        self.register_buffer('t', torch.zeros(num_features))
        raise NotImplementedError("Needs to be updated")

    def forward(self, inputs):
        x, rexp = inputs
        self.t = Round.apply(self.bias[None, :, None, None]*(2**(-rexp)))
        # self.t = self.t.clamp(-128,127)
        if self.training:
            x = x*(2**(-rexp))
            x = x + self.t
            # x = x.clamp(-128,127)
            x = x/(2**(-rexp))
        else:
            x = x + self.t
            # x = x.clamp(-128,127)

        return x, rexp


class BlockQuantN(nn.Module):
    def __init__(self, layers_in, layers_out, kernel_size, stride=1, groups=1, outQuantBits=8, outQuantDyn=False, weight_quant_bits=8, weight_quant_channel_wise=True) -> None:
        super(BlockQuantN, self).__init__()

        self.conv = Conv2dQuant_new(layers_in, layers_out, kernel_size, stride, padding=int(
            np.floor(kernel_size/2)), groups=groups, weight_quant_bits=weight_quant_bits, weight_quant_channel_wise=weight_quant_channel_wise)
        self.bn = BatchNorm2dBase_new(
            layers_out, outQuantBits=outQuantBits, outQuantDyn=outQuantDyn)
        self.activation = LeakReLU(0.125)

    def convert(self):
        return BlockQuantN_(self.conv, self.bn, self.activation)

    def forward(self, invals: Tuple[torch.Tensor, torch.Tensor]):

        fact = self.bn.get_weight_factor()

        x = self.conv(invals, fact)
        x = self.bn(x, self.conv.quantw.delta)
        x = self.activation(x)

        return x


class BlockQuantNwoA(BlockQuantN):
    def __init__(self, layers_in, layers_out, kernel_size, stride=1, groups=1, outQuantBits=8, outQuantDyn=False, weight_quant_bits=8, weight_quant_channel_wise=True) -> None:
        super(BlockQuantNwoA, self).__init__(layers_in, layers_out, kernel_size, stride,
                                             groups, outQuantBits, outQuantDyn, weight_quant_bits, weight_quant_channel_wise)
        self.activation = nn.Sequential()


#########################################################################################
#                                   Common Layers                                       #
#########################################################################################


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

#########################################################################################
#                                   ENCAPSULATED                                        #
#########################################################################################


class MaxPool2d(nn.MaxPool2d):
    def __init__(self, kernel_size: _size_any_t, stride: Optional[_size_any_t] = None, padding:  _size_any_t = 0, dilation:  _size_any_t = 1, return_indices: bool = False, ceil_mode: bool = False) -> None:
        super(MaxPool2d, self).__init__(kernel_size, stride,
                                      padding, dilation, return_indices, ceil_mode)

    def convert(self):
        return nn.MaxPool2d(self.kernel_size, self.stride, self.padding, self.dilation, self.return_indices, self.ceil_mode)

    def forward(self, input: Tuple[torch.Tensor, torch.Tensor]):
        val, rexp = input
        if self.training:
            return (F.max_pool2d(val, self.kernel_size, self.stride,
                                self.padding, self.dilation, self.ceil_mode,
                                self.return_indices),
                    rexp)
        else:
            return (F.max_pool2d(val.type(torch.float32), self.kernel_size, self.stride,
                            self.padding, self.dilation, self.ceil_mode,
                            self.return_indices).type(torch.int32),rexp)
        

