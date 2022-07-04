import torch
import torch.nn as nn
import torch.nn.functional as F

class Round(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class Floor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.floor(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class switch(torch.autograd.Function):
    @staticmethod
    def forward(self, in1, in2):
        return in2, in1

    @staticmethod
    def backward(self, out1: torch.Tensor, out2: torch.Tensor):
        return out1, out2


import numpy as np

class checkNan(torch.autograd.Function):
    @staticmethod
    def forward(self, in1):
        with torch.no_grad():
            if in1.isnan().any():
                print("check nan forward nan")
        return in1

    @staticmethod
    def backward(self, out: torch.Tensor):
        with torch.no_grad():
            if out.isnan().any():
                print("check nan backward nan:",torch.sum(torch.isnan(out)))
                # print(out)
                # npdata = out.view(-1).cpu().detach().numpy()
                # np.savetxt("checkNanvalue.txt",npdata)
            out = out.masked_fill_(torch.isnan(out),0)
        return out.detach()