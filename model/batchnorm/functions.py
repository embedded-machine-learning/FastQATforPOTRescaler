from asyncio.log import logger
import torch
from torch.nn.common_types import Tensor

from ..logger import logger_forward

@logger_forward
def calculate_n(
    weight: Tensor,
    bias: Tensor,
    mean: Tensor,
    var: Tensor,
    out_quant: Tensor,
    rexp: Tensor,
) -> Tensor:
    """
    calculate_n calculates the shift factor

    :param weight: The absolute of the weight vector
    :type weight: Tensor
    :param bias: The bias of the BN
    :type bias: Tensor
    :param mean: The mean of the BN
    :type mean: Tensor
    :param var: The variance of the BN
    :type var: Tensor
    :param out_quant: The out quantization factor
    :type out_quant: Tensor
    :param rexp: The input exponent
    :type rexp: Tensor
    :return: Return the shift factor
    :rtype: Tensor
    """
    with torch.no_grad():
        n = torch.log2(weight.abs() / (out_quant * torch.sqrt(var + 1e-5))) + rexp.view(-1)
        n = n.ceil()
        return n

@logger_forward
def calculate_n_fixed(
    weight: Tensor,
    bias: Tensor,
    mean: Tensor,
    var: Tensor,
    out_quant: Tensor,
    rexp: Tensor,
) -> Tensor:
    """
    calculate_n calculates the shift factor for a whole layer

    :param weight: The absolute of the weight vector
    :type weight: Tensor
    :param bias: The bias of the BN
    :type bias: Tensor
    :param mean: The mean of the BN
    :type mean: Tensor
    :param var: The variance of the BN
    :type var: Tensor
    :param out_quant: The out quantization factor
    :type out_quant: Tensor
    :param rexp: The input exponent
    :type rexp: Tensor
    :return: Return the shift factor
    :rtype: Tensor
    """
    with torch.no_grad():
        n = torch.log2(weight.abs() / (out_quant * torch.sqrt(var + 1e-5))) + rexp.view(-1)
        nr = n.max() * torch.ones_like(n)
        nr = nr.ceil()
        return nr

@logger_forward
def calculate_t(
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    var: torch.Tensor,
    out_quant: torch.Tensor,
    rexp: torch.Tensor,
    n: torch.Tensor,
) -> torch.Tensor:
    """
    calculate_t calculates the additive value

    :param weight: The absolute of the weight vector
    :type weight: Tensor
    :param bias: The bias of the BN
    :type bias: Tensor
    :param mean: The mean of the BN
    :type mean: Tensor
    :param var: The variance of the BN
    :type var: Tensor
    :param out_quant: The out quantization factor
    :type out_quant: Tensor
    :param rexp: The input exponent
    :type rexp: Tensor
    :return: Return the additive value
    :rtype: Tensor
    """
    with torch.no_grad():
        t = -mean * (weight / (out_quant * torch.sqrt(var + 1e-5))) + bias / out_quant
        return t


@logger_forward
def calculate_alpha(
    weight: torch.Tensor,
    mean: torch.Tensor,
    var: torch.Tensor,
    out_quant: torch.Tensor,
    rexp: torch.Tensor,
) -> torch.Tensor:
    """
    calculate_alpha calculates the adaptation factor for the weights

    :param weight: The absolute of the weight vector
    :type weight: Tensor
    :param mean: The mean of the BN
    :type mean: Tensor
    :param var: The variance of the BN
    :type var: Tensor
    :param out_quant: The out quantization factor
    :type out_quant: Tensor
    :param rexp: The input exponent
    :type rexp: Tensor
    :return: The adaptation factor
    :rtype: torch.Tensor
    """
    with torch.no_grad():
        n = weight.abs() * rexp.view(-1)
        n = n.div_(out_quant).div_(torch.sqrt(var + 1e-5))
        n = n.log2_()
        # n = torch.log2(weight.abs() * rexp.view(-1) / (out_quant * torch.sqrt(var + 1e-5)))
        nr = torch.ceil(n)
        alpha = torch.sign(weight) * torch.exp2(n - nr)
    return alpha

@logger_forward
def calculate_alpha_fixed(
    weight: torch.Tensor,
    mean: torch.Tensor,
    var: torch.Tensor,
    out_quant: torch.Tensor,
    rexp: torch.Tensor,
) -> torch.Tensor:
    """
    calculate_alpha calculates the adaptation factor for the weights for a fixed n

    :param weight: The absolute of the weight vector
    :type weight: Tensor
    :param mean: The mean of the BN
    :type mean: Tensor
    :param var: The variance of the BN
    :type var: Tensor
    :param out_quant: The out quantization factor
    :type out_quant: Tensor
    :param rexp: The input exponent
    :type rexp: Tensor
    :return: The adaptation factor
    :rtype: torch.Tensor
    """
    with torch.no_grad():
        n = torch.log2(weight.abs() / (out_quant * torch.sqrt(var + 1e-5))) + rexp.view(-1)
        nr = n.max() * torch.ones_like(n)
        nr = torch.ceil(nr)
        alpha = torch.sign(weight) * torch.exp2(n - nr)
    raise NotImplementedError()
    return alpha

