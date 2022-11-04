import torch
from torch import nn


class Sequential(nn.Sequential):
    def int_extract(self, accumulation_type=torch.int32, small_signed_type=torch.int8, small_unsigned_type=torch.uint8) -> nn.Sequential:
        out = nn.Sequential()
        for module in self:
            tmp = module.int_extract(accumulation_type=accumulation_type, small_signed_type=small_signed_type, small_unsigned_type=small_unsigned_type)
            if tmp is not None:
                out.append(tmp)
        return out
