import torch
from torch.nn.common_types import Tensor

from ...batchnorm import BatchNorm2d
from ...activations import PACT

def test_batchnorm():
    DUT = BatchNorm2d(num_features=100,momentum=1, out_quant=PACT)
    batchnorm = torch.nn.BatchNorm2d(num_features=100,momentum=1)
    act = PACT(8,(1,100,1,1))

    # for i in range(100):
    #     x = torch.rand((10,100,123,45))
    #     DUT_y = DUT(x)
    #     should_be = act(batchnorm(x))

    #     assert torch.isclose(DUT_y,should_be).all()