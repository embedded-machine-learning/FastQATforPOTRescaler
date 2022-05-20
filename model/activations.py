import torch
import torch.nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch.nn.common_types import _size_any_t

from model.utils import *

class LeakReLU(torch.nn.LeakyReLU):
    def __init__(self, negative_slope: float = 0.01, inplace: bool = False) -> None:
        super().__init__(negative_slope, inplace)
    def forward(self,input: Tuple[torch.Tensor,torch.Tensor]):
        x,rexp = input
        x = F.relu(x, inplace=self.inplace)
        if self.training:
            x = x*(2**(-rexp[None,:,None,None]))
            x = Floor.apply(x)
            x = x/(2**(-rexp[None,:,None,None]))
        else:
            x = Floor.apply(x)
        return x,rexp
