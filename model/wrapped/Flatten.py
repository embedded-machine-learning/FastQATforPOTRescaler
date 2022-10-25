from turtle import forward
import torch
import torch.nn as nn

from ..DataWrapper import DataWrapper 

def Flatten(input: DataWrapper, dim: int) -> DataWrapper:
    """
    Flatten encapsulation of torch.flatten
    """
    val, rexp = input.get()
    orexp = rexp.detach() * torch.ones_like(val[0, ...]) # creates a 
    return input.set(val.flatten(dim), orexp.flatten(dim))

class FlattenM(nn.Module):
    def __init__(self,dim) -> None:
        super(FlattenM,self).__init__()
        self.dim = dim
    def forward(self,input:DataWrapper) -> DataWrapper:
        return Flatten(input,self.dim)