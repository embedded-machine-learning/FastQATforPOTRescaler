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
        return grad_input

class switch(torch.autograd.Function):
    @staticmethod
    def forward(self, in1, in2):
        return in2, in1

    @staticmethod
    def backward(self, out1: torch.Tensor, out2: torch.Tensor):
        return out1.detach(), out2.detach()

running_exponent=0

def set_rexp(val):
    global running_exponent
    running_exponent = val
def get_rexp():
    global running_exponent
    return running_exponent

