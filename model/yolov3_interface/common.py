import torch

from ..blocks import ConvBn, ConvBnA
from ..activations import PACT



def ConvQAT(c1, c2, k=1, s=1, p=None, g=1, act=True):
    if act:
        return ConvBnA(
            in_channels=c1,
            out_channels=c2,
            kernel_size=k,
            stride=s,
            padding=p,
            groups=g,
            activation=PACT
        )
    return ConvBn(
        in_channels=c1,
        out_channels=c2,
        kernel_size=k,
        stride=s,
        padding=p,
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
        rexp = torch.cat(tuple([e if len(e.shape)>1 else e.view(1).expand(q.shape[1]).view(1,-1,1,1) for q in rexp]),self.dim)
        return x[0].set(vals,rexp)