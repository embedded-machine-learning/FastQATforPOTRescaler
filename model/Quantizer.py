# Generic Type imports
from typing import Optional, Tuple, Union

# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_any_t, _size_2_t, Tensor, _size_any_opt_t



def FakeQuant(
    x: Tensor,
    delta_in: Tensor,
    delta_out: Tensor,
    training: bool,
    min_quant: Tensor,
    max_quant: Tensor,
    rounding_mode: str = "floor",
) -> Tensor:
    with torch.no_grad():
        if training:
            x.data.div_(delta_in, rounding_mode=rounding_mode).clamp_(min_quant, max_quant).mul_(delta_out)
        else:
            x = x.data.div(delta_in, rounding_mode=rounding_mode).clamp_(min_quant, max_quant)
    return x
