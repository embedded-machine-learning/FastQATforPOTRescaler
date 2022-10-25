from torch import nn

from ..logger import logger_forward,logger_init
from ..DataWrapper import DataWrapper


class Dropout(nn.Dropout):
    """
    Dropout Wrapped Dropout

    Hey no idea why you would want dropout in QAT but here you go

    :param p: Chance, defaults to 0.5
    :type p: float, optional
    """
    @logger_init
    def __init__(self, p: float = 0.5) -> None:
        super(Dropout,self).__init__(p, False)
    
    @logger_forward
    def forward(self,input:DataWrapper)->DataWrapper:
        x,rexp = input.get()
        return input.set(super(Dropout,self).forward(x),rexp)