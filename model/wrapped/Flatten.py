import torch
import torch.nn as nn

from ..Type import Data_wrapper 

def Flatten(input: Data_wrapper, dim: int) -> Data_wrapper:
    """
    Flatten encapsulation of torch.flatten
    """
    val, rexp = input.get()
    orexp = rexp.detach() * torch.ones_like(val[0, ...]) # creates a 
    return input.set(val.flatten(dim), orexp.flatten(dim))