import torch
from torch import nn
from torch.nn.common_types import Tensor
import torch.nn.functional as F


class Linear_int(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        quant_weights: Tensor,
        shift: Tensor,
        bias: Tensor,
        Act_min: Tensor,
        Act_max: Tensor,
    ) -> None:
        super().__init__(in_features, out_features, False)
        self.weight.requires_grad_(False)
        self.weight.data = quant_weights
        self.register_buffer("n", shift)
        if bias is not None:
            self.register_buffer("t", bias)
        else:
            self.t = None
        self.register_buffer("min", Act_min)
        self.register_buffer("max", Act_max)

    def forward(self, x: Tensor) -> Tensor:
        out = F.linear(x, self.weight, None)
        out = torch.bitwise_right_shift(out, -self.n)
        if self.t is not None:
            out = out + self.t
        out = out.clamp(self.min,self.max)
        return out