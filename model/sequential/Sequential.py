import torch
from torch import nn


class Sequential(nn.Sequential):
    def int_extract(self, type_small=torch.int8, type_big=torch.int32) -> nn.Sequential:
        out = nn.Sequential()
        for module in self:
            tmp = module.int_extract(type_small=type_small, type_big=type_big)
            if tmp is not None:
                out.append(tmp)
        return out
