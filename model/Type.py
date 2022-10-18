# Generic Type imports
from typing import Optional, Tuple, Union

# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_any_t, _size_2_t, Tensor, _size_any_opt_t


class Data_wrapper:
    value = torch.empty((1))
    rexp = torch.empty((1))

    rexp_force = torch.empty((1))

    to_copy = [
        'rexp_force',
    ]

    def __init__(self,value=None,rexp=None) -> None:
        super(Data_wrapper, self).__init__()
        self.value = value
        self.rexp = rexp

    def __repr__(self)-> str:
        return "value:{0}, rexp:{1}, rexp_force:{2}".format(self.value,self.rexp,self.rexp_force)

    def copy(self,other:'Data_wrapper'):
        for key in self.to_copy:
            setattr(self,key,getattr(other,key))
        return self

    def get(self) -> Tuple[Tensor, Tensor]:
        return self.value, self.rexp

    def set(self,value:Tensor,rexp:Tensor):
        self.value = value
        self.rexp = rexp
        return self

    