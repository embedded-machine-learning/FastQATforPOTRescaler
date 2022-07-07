import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Union, Tuple

from .quantizer import *
from .utils import *


class Conv2d_(nn.Conv2d):
    def __init__(self, weights, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1, padding: Union[str, _size_2_t] = 0, dilation: _size_2_t = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', device=None, dtype=None) -> None:
        super(Conv2d_, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        self.weight = nn.Parameter(weights.clone(), requires_grad=True)
        self.register_buffer("used_weights", weights.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.used_weights = Round.apply(self.weight)
        self.used_weights = self.used_weights.clamp(-128, 127)
        return self._conv_forward(x, self.used_weights, None)


class Conv2dQuant(nn.Conv2d):
    def __init__(self,  in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1,
                 padding: Union[str, _size_2_t] = 0, dilation: _size_2_t = 1, groups: int = 1,
                 bias: bool = False, padding_mode: str = 'zeros', device=None, dtype=None,
                 weight_quant=None, weight_quant_bits=None, weight_quant_channel_wise=False, weight_quant_args=None, weight_quant_kargs={}) -> None:
        super(Conv2dQuant, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias, padding_mode, device, dtype)

        if weight_quant_args == None:
            quant_qrgs = (8 if weight_quant_bits == None else weight_quant_bits,
                          (-1,) if not weight_quant_channel_wise else (out_channels, 1, 1, 1),
                          0.1
                          )
        else:
            quant_qrgs = weight_quant_args

        if weight_quant == None:
            self.quantw = LinQuant(*quant_qrgs, **weight_quant_kargs)
        else:
            self.quantw = weight_quant

        self.register_buffer('quant_weight', torch.zeros_like(self.weight))

    def convert(self):
        return Conv2d_(self.used_weights, self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias, self.padding_mode, None, None)

    def forward(self, invals: Tuple[torch.Tensor, torch.Tensor], factor=1) -> torch.Tensor:
        input, rexp = invals

        tmp = self.weight.clone()

        orexp = (torch.mean(rexp)).squeeze()
        rexp_diff = rexp.squeeze() - orexp.unsqueeze(-1)
        tmp = tmp*(2**rexp_diff)[None, :, None, None]

        tmp = self.quantw(tmp, factor)

        if not self.training:
            tmp = torch.round(tmp/self.quantw.delta)
            # only nessesary as /delta can have a slight relative error ~1e-6 in calculations
            self.quant_weight = tmp.detach()
        else:
            tmp = tmp/((2**rexp_diff)[None, :, None, None])

        if torch.any(torch.isnan(tmp)):
            print(torch.max(torch.abs(self.weight.view(-1))))
            print(factor)


        out = self._conv_forward(input, tmp, None)
        if torch.any(torch.isnan(out-out.round())):
            print("WTF")
        if not self.training and (out-out.round()).abs().max()!=0:
            print("post convolution not whole number",(out-out.round()).mean())
        return out, orexp

    



class Conv2dQuant_new(nn.Conv2d):
    def __init__(self,  in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1,
                 padding: Union[str, _size_2_t] = 0, dilation: _size_2_t = 1, groups: int = 1,
                 bias: bool = False, padding_mode: str = 'zeros', device=None, dtype=None,
                 weight_quant=None, weight_quant_bits=None, weight_quant_channel_wise=False, weight_quant_args=None, weight_quant_kargs={}) -> None:
        super(Conv2dQuant_new, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias, padding_mode, device, dtype)

        if weight_quant_args == None:
            quant_qrgs = (8 if weight_quant_bits == None else weight_quant_bits,
                          (-1,) if not weight_quant_channel_wise else (out_channels, 1, 1, 1),
                          1
                          )
        else:
            quant_qrgs = weight_quant_args

        if weight_quant == None:
            self.quantw = LinQuant(*quant_qrgs, **weight_quant_kargs)
        else:
            self.quantw = weight_quant

        self.register_buffer('quant_weight', torch.zeros_like(self.weight))

    def convert(self):
        return Conv2d_(self.used_weights, self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias, self.padding_mode, None, None)

    def forward(self, invals: Tuple[torch.Tensor, torch.Tensor], factor=1) -> torch.Tensor:
        input, rexp = invals

        tmp = self.weight.clone()

        orexp = (torch.mean(rexp)).squeeze()
        rexp_diff = rexp.squeeze() - orexp.unsqueeze(-1)
        tmp = tmp*(2**rexp_diff)[None, :, None, None]

        tmp = self.quantw(tmp, factor)/factor

        if not self.training:
            tmp = torch.round(tmp/self.quantw.delta*factor)
            # only nessesary as /delta can have a slight relative error ~1e-6 in calculations
            self.quant_weight = tmp.detach()
        else:
            tmp = tmp/((2**rexp_diff)[None, :, None, None])

        if torch.any(torch.isnan(tmp)):
            print(torch.max(torch.abs(self.weight.view(-1))))
            print(factor)


        out = self._conv_forward(input, tmp, None)
        # if torch.any(torch.isnan(out-out.round())):
        #     print("WTF")
        # if not self.training and (out-out.round()).abs().max()!=0:
        #     print("post convolution not whole number",(out-out.round()).mean())
        return out, orexp
