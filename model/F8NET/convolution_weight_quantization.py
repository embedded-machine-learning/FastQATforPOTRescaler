import torch
from torch.nn.common_types import Tensor

from types import FunctionType
from typing import Tuple

from ..Quantizer import FakeQuant


from ..convolution.weight_quantization import LinQuantWeight 

from ..logger import logger_init, logger_forward


class LinQuantWeight_mod_F8NET(LinQuantWeight):
    @logger_init
    def __init__(self, bits: int = 8, size: tuple = (-1,), rounding_mode: str = "trunc", layer_wise=False) -> None:
        super().__init__(bits, size, rounding_mode,layer_wise)
        self.register_buffer("delta_in_factor", torch.tensor(1.0 / 40.0))
        self.register_buffer("delta_out_factor", torch.tensor(1.0 / 40.0))

    @logger_forward
    def forward(self, x: Tensor, rexp_mean: Tensor, rexp_diff: Tensor, fact_fun: FunctionType) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            sigma = (
                torch.var(x * (rexp_diff.view(*self.rexp_view)), self.reducelist, unbiased=False, keepdim=True)
                .add_(1e-5)
                .sqrt_()
            )

            self.delta_in = sigma.mul_(self.delta_in_factor)  # delta in and delta out identical
            self.delta_out.data = self.delta_in

            fact = fact_fun(self.delta_out * rexp_mean.view(-1,1,1,1)).view(-1, 1, 1, 1)

            delta_for_quant = self.delta_in.div(rexp_diff.view(*self.rexp_view)).div_(fact)

        return (
            FakeQuant(
                x=x.clone(),
                delta_in=delta_for_quant,
                delta_out=delta_for_quant,
                training=self.training,
                min_quant=self.min,
                max_quant=self.max,
                rounding_mode=self.rounding_mode,
            ),
            fact,
        )