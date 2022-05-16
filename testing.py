from tracemalloc import start
import torch
import torch.nn
from model.quantizer import *
from model.layer import *
from model.convolution import *

a = torch.zeros(2,3,10,10)

st = Start(-8)
c1 = Conv2dLinChannelQuant(3,16,3)

x = st(a)

print(x[0].shape,x[1].shape)
x = c1(x)
print(x[0].shape,x[1].shape)


tmp=nn.Sequential(
    Start(-8),
    Conv2dLinChannelQuant(3,16,3),
    Conv2dExpLayerQuant(16,32,3),
    Stop()
)
tmp(a)

print(tmp(a).shape)
