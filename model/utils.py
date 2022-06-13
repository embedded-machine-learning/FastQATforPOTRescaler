import torch
import torch.nn as nn
import torch.nn.functional as F

class Round(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        # print("Round grad max :", torch.max(grad_input.view(-1)))
        return grad_input

class Floor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return torch.floor(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        # print("Floor grad max :", torch.max(grad_input.view(-1)))
        return grad_input

class switch(torch.autograd.Function):
    @staticmethod
    def forward(self, in1, in2):
        return in2, in1

    @staticmethod
    def backward(self, out1: torch.Tensor, out2: torch.Tensor):
        return out1.detach(), out2.detach()


import numpy as np

class checkNan(torch.autograd.Function):
    @staticmethod
    def forward(self, in1):
        if torch.any(torch.isnan(in1)):
            print("check nan forward nan")
        return in1

    @staticmethod
    def backward(self, out: torch.Tensor):
        if torch.any(torch.isnan(out)):
            print("check nan backward nan:",torch.sum(torch.isnan(out)))
            # print(out)
            # npdata = out.view(-1).cpu().detach().numpy()
            # np.savetxt("checkNanvalue.txt",npdata)
        out = out.masked_fill(torch.isnan(out),0)
        return out.detach()