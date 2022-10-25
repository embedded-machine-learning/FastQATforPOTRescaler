import torch
from torch import nn

from ..DataWrapper import DataWrapper
from ..logger import logger_forward, logger_init


def Flatten(input: DataWrapper, dim: int) -> DataWrapper:
    """
    Flatten encapsulation of torch.flatten
    """
    val, rexp = input.get()
    orexp = rexp.detach() * torch.ones_like(val[0, ...])  # creates a
    return input.set(val.flatten(dim), orexp.flatten(dim))


class FlattenM(nn.Module):
    """
    FlattenM A simple module to use flatten as a layer

    :param dim: The flatten dimension
    :type dim: int
    """

    @logger_init
    def __init__(self, dim: int) -> None:
        super(FlattenM, self).__init__()
        self.dim = dim

    @logger_forward
    def forward(self, input: DataWrapper) -> DataWrapper:
        return Flatten(input, self.dim)
