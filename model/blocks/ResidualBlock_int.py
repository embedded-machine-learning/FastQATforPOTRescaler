from typing import Any, Optional, Union

from torch import Tensor, nn

from .ConvBnA_int import ConvBnA_int
from ..add.Add_int import Add_int

from ..logger import logger_forward,logger_init

class ResidualBlock_int(nn.Module):
    """
    ResidualBlock_int The Residual Block from Resnet in pure integer

    :param block1: The first block of the Residual Block
    :type block1: ConvBnA_int
    :param block2: The second block of the Residual Block
    :type block2: ConvBnA_int
    :param down_sample_block: The down sample block of the Residual Block, if it exists, otherwise use None
    :type down_sample_block: Optional[Union[ConvBnA_int,Any]]
    :param add: The add layer
    :type add: Add_int
    """
    @logger_init
    def __init__(
        self,
        block1 : ConvBnA_int,
        block2 : ConvBnA_int,
        down_sample_block: Optional[Union[ConvBnA_int,Any]],
        add : Add_int
    ) -> None:
        super(ResidualBlock_int,self).__init__()
        self.block1 = block1
        self.block2 = block2
        self.down_sample_block = down_sample_block
        self.add = add

    @logger_forward
    def forward(self,x:Tensor)->Tensor:
        bypass = x.clone()
        out = self.block1(x)
        out = self.block2(out)
        if self.down_sample_block is not None:
            bypass = self.down_sample_block(bypass)
        out = self.add(out,bypass)
        return out