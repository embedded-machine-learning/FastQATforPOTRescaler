from turtle import forward
import torch
import torch.nn as nn

from ..logger import logger_forward,logger_init
from ..DataWrapper import DataWrapper


class Dropout(nn.Dropout):
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super(Dropout,self).__init__(p, False)
    
    def forward(self,input:DataWrapper)->DataWrapper:
        x,rexp = input.get()
        return input.set(super(Dropout,self).forward(x),rexp)