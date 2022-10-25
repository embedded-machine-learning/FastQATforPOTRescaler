# Generic Type imports
from typing import Optional, Tuple, Union, Any

# Torch imports
import torch
from torch.nn.common_types import Tensor


class DataWrapper:
    """
    DataWrapper is the standard Data Type passed by layers

    It contains the value, the exponential factor (elem. R) with a base of 2.
    There is also meta data contained such as a quantization base 'quant_val'

    """
    value = torch.empty((1))
    rexp = torch.empty((1))


    quant_val = None

    to_copy = [
        'quant_val'
    ]

    def __init__(self,value=None,rexp=None) -> None:
        super(DataWrapper,self).__init__()
        self.value = value
        self.rexp = rexp

    def __repr__(self)-> str:
        return f"value:{self.value}, rexp:{self.rexp}, quant_fnc:{self.quant_val}"

    def copy(self,other:'DataWrapper') -> 'DataWrapper':
        """
        copy copies other to current

        :param other: another DataWrapper
        :type other: DataWrapper
        :return: self
        :rtype: DataWrapper
        """
        for key in self.to_copy:
            setattr(self,key,getattr(other,key))
        return self

    def get(self) -> Tuple[Tensor, Tensor]:
        """
        gets value and rexp

        :return: a Tuple containing the values (value, rexp)
        :rtype: Tuple[Tensor, Tensor]
        """
        return self.value, self.rexp

    def set(self,value:Tensor,rexp:Tensor) -> 'DataWrapper':
        """
        set creates a new object and adds the value and rexp to it, then copies the meta data

        :param value: The value
        :type value: Tensor
        :param rexp: The exponent
        :type rexp: Tensor
        :return: self
        :rtype: DataWrapper
        """
        return DataWrapper(value,rexp).copy(self)

    def __getitem__(self, key: str) -> Any:
        """
        __getitem__ adds the possibility to access internal data by bracket operation

        :param key: the name of the parameter
        :type key: str
        :return: the parameter
        :rtype: Any
        """
        return getattr(self,key)

    
    def set_quant(self,value:Optional[Union[Tensor,'DataWrapper']]=None):
        """
        set_quant Sets the quantization level

        Use if a quantitation level needs to be enforced later 

        :param value: Value to be used as base, if None use current rexp, defaults to None
        :type value: Optional[Tensor], optional
        """
        if value is None:
            value = self.rexp
        if isinstance(value,DataWrapper):
            value = value.rexp
        self.quant_val = value.clone().detach().exp2()

    def use_quant(self,x:Tensor)->Tensor:
        """
        use_quant enforces a quantization level as a power of 2 multitude of 'quant_val'

        :param x: The to standardize value
        :type x: Tensor
        :return:  The standardize value
        :rtype: Tensor
        """
        if self.quant_val is None:
            return x
        return x.div(self.quant_val).log2().round().exp2().mul(self.quant_val)
    