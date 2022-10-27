import torch
from torch import nn


class Sequential(nn.Sequential):
    def int_extract(self) -> nn.Sequential:
        out = nn.Sequential()
        for module in self:
            tmp = module.int_extract()
            if tmp is not None:
                out.append(tmp)
        return out
