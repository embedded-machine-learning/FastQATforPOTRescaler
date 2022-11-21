import torch
import torch.nn as nn

from ..blocks import ConvBn, ConvBnA
from ..activations import PACT

# taken from yolov3
def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

def ConvQAT(c1, c2, k=1, s=1, p=None, g=1, act=True):
    if act:
        return ConvBnA(
            in_channels=c1,
            out_channels=c2,
            kernel_size=k,
            stride=s,
            padding=autopad(k, p),
            groups=g,
            activation=PACT,
        )
    return ConvBn(
        in_channels=c1,
        out_channels=c2,
        kernel_size=k,
        stride=s,
        padding=autopad(k, p),
        groups=g,
    )

class ConcatQAT(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.dim = dimension

    def forward(self, x):
        vals = [t.get()[0] for t in x]
        rexp = [t.get()[1] for t in x]
        vals = torch.cat(tuple(vals),self.dim)
        rexp = torch.cat(tuple([inp.get()[1] if len(inp.get()[1].shape)>1 else inp.get()[1].view(1).expand(inp.get[0].shape[1]).view(1,-1,1,1) for inp in x]),self.dim)
        return x[0].set(vals,rexp)