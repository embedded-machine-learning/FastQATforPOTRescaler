import torch
import torch.nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch.nn.common_types import _size_any_t

from .utils import *

class LeakReLU_(torch.nn.LeakyReLU):
    def __init__(self, negative_slope: float = 0.01, inplace: bool = False) -> None:
        super(LeakReLU_,self).__init__(negative_slope, inplace)
    def forward(self,x: torch.Tensor):
        x = F.leaky_relu(x,negative_slope=self.negative_slope,inplace=self.inplace)
        x = Floor.apply(x)
        return x

class LeakReLU(torch.nn.LeakyReLU):
    def __init__(self, negative_slope: float = 0.01, inplace: bool = False) -> None:
        super().__init__(negative_slope, inplace)
    def forward(self,input: Tuple[torch.Tensor,torch.Tensor]):
        x,rexp = input
        x = F.leaky_relu(x,negative_slope=self.negative_slope, inplace=self.inplace)
        if self.training:
            x = x*(2**(-rexp))
            x = Floor.apply(x)
            x = x/(2**(-rexp))
        else:
            x = Floor.apply(x)
        return x,rexp

    def convert(self):
        return LeakReLU_(self.negative_slope,self.inplace)
