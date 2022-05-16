import torch
import torch.nn as nn
import torch.nn.functional as F

from model_old.quantizer import *

class PReLUQuant(nn.PReLU):     # TODO QUANTIZE
    def __init__(self, num_parameters: int = 1, init: float = 0.25, device=None, dtype=None) -> None:
        super(PReLUQuant, self).__init__(num_parameters, init, device, dtype)
        self.register_buffer('zero_mask', torch.zeros_like(
            self.weight, dtype=torch.bool))
        self.register_buffer('exp_vals', torch.zeros_like(
            self.weight, dtype=torch.int))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # TODO ausbessern so das kein nan entstehen kann
        #        if self.training:
        tmp, self.exp_vals, self.zero_mask = specialExpQuant.apply(self.weight)
        # else:
        #   tmp = torch.exp2(self.exp_vals)
        #   tmp.masked_fill(~self.zero_mask, 0)
        return F.prelu(input, tmp)