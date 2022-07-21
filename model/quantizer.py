import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import *


#################################################
#           FUNCTIONS                           #
#################################################

class expQuant(torch.autograd.Function):
    @staticmethod
    def forward(self, x):
        tmp = torch.exp2(torch.ceil(torch.log2(x)))
        if torch.any(torch.isnan(tmp)):
            print("nan in exp scale")
        return tmp

    @staticmethod
    def backward(self, grad_output):
        if torch.any(torch.isnan(grad_output)):
            print("nan in exp scale")
        return grad_output.detach()


class LinQuant_(torch.autograd.Function):
    @staticmethod
    def forward(self, x, abs, delta, bits):
        with torch.no_grad():
            self.save_for_backward(x, abs)
            # x = x.clamp(-abs, abs)
            x = x.div(delta,rounding_mode = "floor").clamp(-2**(bits-1),2**(bits-1)-1).mul(delta)
            if torch.any(torch.isnan(x)):
                print("nan in Linquant forward")
            return x
        
    @staticmethod
    def backward(self, grad_output: torch.Tensor):
        with torch.no_grad():
            val = 2
            x, abs = self.saved_tensors
            grad_output = grad_output.masked_fill(torch.logical_and(
                torch.gt(x, val*abs), torch.gt(grad_output, 0)), 0)
            grad_output = grad_output.masked_fill(torch.logical_and(
                torch.le(x, -val*abs), torch.le(grad_output, 0)), 0)
            if torch.any(torch.isnan(grad_output)):
                print("nan in Linquant back")
            return grad_output.detach(), None, None, None


class specialExpQuant(torch.autograd.Function):
    def forward(self, x: torch.Tensor):
        self.save_for_backward(x)
        x = x.clamp(0, 1)
        cond = torch.gt(x, 2**-6)
        n = torch.where(cond, torch.log2(
            expQuant.apply(x)), torch.zeros_like(x))
        x = torch.where(cond, expQuant.apply(x), torch.zeros_like(x))
        if torch.any(torch.isnan(x)):
            print("nan in specialExpQuant forward")
        return x.detach(), n.detach(), cond.detach()

    def backward(self, grad_output: torch.Tensor, n_output: torch.Tensor, cond_output: torch.Tensor):
        x, = self.saved_tensors

        # only allow positive graients when greater 1 so that it can still get smaller
        grad_output = grad_output.masked_fill(torch.logical_and(
            torch.gt(x, 2.0), torch.gt(grad_output, 0)), 0)
        # only allow negative graients when smaller 2**-6 so that it can still get bigger
        grad_output = grad_output.masked_fill(torch.logical_and(
            torch.le(x, (2**-7)*1.0), torch.le(grad_output, 0)), 0)
        if torch.any(torch.isnan(grad_output)):
            print("nan in specialExpQuant backward")
        return grad_output.detach(), None, None


def get_abs(self,x:torch.Tensor)->torch.Tensor:
    if self.simple:
        abs = x.abs().max()
    else:
        xreorderd = x.permute(self.permutelist).contiguous()
        xreorderd = xreorderd.view((*xreorderd.shape[:self.numberofdims],-1))
        abs = xreorderd.abs().max(dim=(self.numberofdims), keepdim=True).values.view(self.size)
    return abs

def get_mean(self,x:torch.Tensor)->torch.Tensor:
    if self.simple:
        abs = x.abs().mean()
    else:
        xreorderd = x.permute(self.permutelist).contiguous()
        xreorderd = xreorderd.view((*xreorderd.shape[:self.numberofdims],-1))
        abs = xreorderd.abs().mean(dim=(self.numberofdims), keepdim=True).view(self.size)
    return abs


#################################################
#           MODULES                             #
#################################################
class Filter_max(nn.Module):
    def __init__(self) -> None:
        super(Filter_max,self).__init__()
        self.past=[]
        self.epoch_length=-1
        self.last_training=False
        self.i=0 
        self.last_dtype = None
        self.dev = None

    def forward(self,x:torch.Tensor):
        # return x
        mult = 1
        with torch.no_grad():
            if self.training:
                if self.last_dtype!=x.dtype or x.device!=self.dev:
                    print("Reseting Filter")
                    self.last_dtype = x.dtype
                    self.past=[]
                    self.epoch_length=-1
                    self.i=0
                    self.dev=x.device

                if not self.last_training:
                    self.last_training=True
                    self.i=0
                
                if self.epoch_length==-1:
                    self.past.append(x.view(-1).detach().clone())
                else:
                    if self.i >= self.epoch_length:
                        print("incorrect epoch length")
                        self.past.append(x.view(-1).detach().clone())
                        self.epoch_length=-1
                    else: 
                        self.past[self.i]=x.view(-1).detach().clone()
                        pass
                # if (self.epoch_length==-1 and self.i==0) or self.epoch_length==1:
                #     out = self.past[0]
                # else:
                out = torch.max(torch.stack(self.past,dim=1),dim=1,keepdim=True).values
                # print(out.shape)
                # print(x.shape)
                x.data = out.detach().clone().view(x.shape).to(x.device).type(x.dtype)
                self.i += 1
            else:
                if self.last_training:
                    self.last_training=False
                    if self.epoch_length==-1:
                        self.epoch_length=self.i
                        print("fixated length ")
                out = torch.max(torch.stack(self.past,dim=1),dim=1).values
                x.data = out.detach().clone().view(x.shape).to(x.device).type(x.dtype)
            return mult*x

class Filter_mean(nn.Module):
    def __init__(self) -> None:
        super(Filter_mean,self).__init__()
        self.past=[]
        self.epoch_length=-1
        self.last_training=False
        self.i=0 
        self.last_dtype = None
        self.dev = None

    def forward(self,x:torch.Tensor):
        # return x
        mult = 1
        x = x.type(torch.float)
        with torch.no_grad():
            if self.training:
                if self.last_dtype!=x.dtype or x.device!=self.dev:
                    print("Reseting Filter")
                    self.last_dtype = x.dtype
                    self.past=[]
                    self.epoch_length=-1
                    self.i=0
                    self.dev=x.device

                if not self.last_training:
                    self.last_training=True
                    self.i=0
                
                if self.epoch_length==-1:
                    self.past.append(x.view(-1).detach().clone())
                else:
                    if self.i >= self.epoch_length:
                        print("incorrect epoch length")
                        self.past.append(x.view(-1).detach().clone())
                        self.epoch_length=-1
                    else: 
                        self.past[self.i]=x.view(-1).detach().clone()
                        pass
                # if (self.epoch_length==-1 and self.i==0) or self.epoch_length==1:
                #     out = self.past[0]
                # else:
                out = torch.mean(torch.stack(self.past,dim=1),dim=1,keepdim=True)
                # print(out.shape)
                # print(x.shape)
                x.data = out.detach().clone().view(x.shape).to(x.device).type(x.dtype)
                self.i += 1
            else:
                if self.last_training:
                    self.last_training=False
                    if self.epoch_length==-1:
                        self.epoch_length=self.i
                        print("fixated length ")
                out = torch.mean(torch.stack(self.past,dim=1),dim=1)
                x.data = out.detach().clone().view(x.shape).to(x.device).type(x.dtype)
            return mult*x

class Quant(nn.Module):
    def __init__(self, size) -> None:
        super(Quant, self).__init__()
        self.simple = False
        if size == (-1,):
            self.register_buffer('delta', torch.ones(1))
            self.size = (1,)
            self.simple = True
        else:
            self.register_buffer('delta', torch.ones(size))
            self.size = size

        self.permutelist = []
        self.numberofdims = 0
        for i in range(len(size)):
            # print(size[i])
            if size[i]!=1:  
                self.permutelist.insert(0,i)
                self.numberofdims += 1
            else:
                self.permutelist.append(i)
        self.permutelist = tuple(self.permutelist)

    def forward(self, x):
        raise NotImplementedError()


class LinQuant(Quant):
    def __init__(self, bits, size=(-1,), mom1=0.1, mom2=0.01) -> None:
        super(LinQuant, self).__init__(size)
        self.bits = bits
        if size == (-1,):
            self.register_buffer('abs', torch.ones(1))
        else:
            self.register_buffer('abs', torch.ones(size))
        self.take_new = True
        self.mom1 = mom1
        self.mom2 = mom2

        self.filter = Filter_max()

    def forward(self, x:torch.Tensor, fact=1):
        with torch.no_grad():
            abs = get_abs(self,x)
            # print(abs)
            # if torch.any(abs < 1e-6):
            #     print("weights to small to quantize")
            #     self.delta = (2*(self.abs/(2.0**self.bits-1))).detach()
            #     return LinQuant_.apply(x*fact, self.abs, self.delta, self.bits)

            # if self.training and self.take_new:
            #     self.abs = abs.detach()
            #     self.take_new = False
            #     # print("new taken")
            # elif self.training:
            #     # print(f" abs diff:  {(abs.view(-1)-self.abs.view(-1)).abs().max()}")
            #     self.abs = ((1-self.mom1-self.mom2)*self.abs + self.mom1*abs + self.mom2 *
            #                 (self.abs/(2.0**self.bits-1.0)) * (2.0**self.bits-1.0)).detach()
            # self.abs = self.filter(abs).detach().clone()
            self.abs = ((1-self.mom1-self.mom2)*self.abs + self.mom1*abs).detach()
            # print(f" old delta: {self.delta.view(-1)}")
            self.delta = (2*(self.abs/(2.0**self.bits-1))).detach()
            # print(f" new delta: {self.delta.view(-1)}")
        # print((self.delta).shape)
        return LinQuant_.apply(x*fact, self.abs, self.delta, self.bits)

class LinQuantExpScale(Quant):
    def __init__(self, bits, size=(-1,), mom1=0.1, mom2=0.01) -> None:
        super(LinQuantExpScale, self).__init__(size)
        self.bits = bits
        if size == (-1,):
            self.register_buffer('abs', torch.ones(1))
        else:
            self.register_buffer('abs', torch.ones(size))
        self.take_new = True
        self.mom1 = mom1
        self.mom2 = mom2

        self.filter = Filter_mean()

    def forward(self, x):
        mult = 8
        with torch.no_grad():
            # abs = mult*get_mean(self,x)
            abs = get_abs(self,x)
            # print(abs.view(-1))
            # print(get_abs(self,x))
            # exit(-1)
            if torch.any(abs < 1e-6):
                print("weights to small to quantize")
                self.delta = (
                    2*expQuant.apply(2/(2.0**self.bits-1.0))).detach()
                return LinQuant_.apply(x, expQuant.apply(10), self.delta, self.bits)
            
            # print(abs)

            if self.training and self.take_new:
                self.abs = abs.detach()
                self.take_new = False
            elif self.training:
                self.abs = ((1-self.mom1-self.mom2)*self.abs + self.mom1*abs + self.mom2 *
                            expQuant.apply(2*self.abs/(2.0**self.bits-1.0)) * (2.0**self.bits-1.0)/2).detach()

            # abs = self.filter(abs).detach().clone()
            # self.abs = (expQuant.apply(self.abs/(2.0**self.bits-1.0))* (2.0**self.bits-1.0)).detach()

            self.delta = (expQuant.apply(2*self.abs/(2.0**self.bits-1.0))).detach()

        # print((self.delta).shape)
        return LinQuant_.apply(x, (expQuant.apply(2*self.abs/(2.0**self.bits-1.0))* (2.0**self.bits-1.0))/2, self.delta, self.bits)


#########################################################################################################################
#                                    NEW STUFF                                                                          #
#########################################################################################################################

class FakeQuant(torch.autograd.Function):
    @staticmethod
    def forward(self,x,factor):
        with torch.no_grad():
            return factor*torch.floor(x/factor)
    def backward(self,grad):
        return grad , None


class LinQuant_new_(torch.autograd.Function):
    @staticmethod
    def forward(self, x, abs, delta,rexp_diff,fact):
        with torch.no_grad():
            x = x*(2**rexp_diff)[None, :, None, None]*fact
            self.save_for_backward(x, abs)
            x = x.clamp(-abs, abs)
            x = x.div(delta, rounding_mode="floor").mul(delta)
            x = x/((2**rexp_diff)[None, :, None, None]*fact)
            if torch.any(torch.isnan(x)):
                print("nan in Linquant forward")
            return x
        
    @staticmethod
    def backward(self, grad_output: torch.Tensor):
        with torch.no_grad():
            val = 1
            x, abs = self.saved_tensors
            grad_output = grad_output.masked_fill(torch.logical_and(
                torch.gt(x, val*abs), torch.gt(grad_output, 0)), 0)
            grad_output = grad_output.masked_fill(torch.logical_and(
                torch.le(x, -val*abs), torch.le(grad_output, 0)), 0)
            if torch.any(torch.isnan(grad_output)):
                print("nan in Linquant back")
            return grad_output.detach(), None, None, None, None


class LinQuantWeight(Quant):
    def __init__(self, bits, size=(-1,), mom1=0.1, mom2=0.01) -> None:
        super(LinQuantWeight, self).__init__(size)
        self.bits = bits
        if size == (-1,):
            self.register_buffer('abs', torch.ones(1))
        else:
            self.register_buffer('abs', torch.ones(size))
        self.take_new = True
        self.mom1 = mom1
        self.mom2 = mom2

    def forward(self, x:torch.Tensor,rexp_diff, fact_fun=None):
        with torch.no_grad():
            abs = get_abs(self,x*(2**rexp_diff.view(-1))[None, :, None, None])
            if torch.any(abs < 1e-6):
                print("weights to small to quantize")
                self.delta = (2*(self.abs.type(abs.dtype)/(2.0**self.bits.type(abs.dtype)-1.0))).detach().type(abs.dtype)
                if fact_fun!=None:
                    fact = fact_fun(self.delta).view(-1,1,1,1)
                else:
                    fact = 1
                if torch.any(torch.isnan(self.delta)):
                    print("nan in weights")
                # print((self.delta).shape)
                return LinQuant_new_.apply(x, self.abs, self.delta,rexp_diff,fact),fact

               
            self.abs = abs.detach()
            # print(f" old delta: {self.delta.view(-1)}")
            self.delta = (2*(self.abs/(2.0**self.bits-1.0))).detach()
            # print(f" new delta: {self.delta.view(-1)}")
            if fact_fun!=None:
                fact = fact_fun(self.delta).view(-1,1,1,1)
            else:
                fact = 1
            if torch.any(torch.isnan(self.delta)):
                print("nan in weights")
        # print((self.delta).shape)
        return LinQuant_new_.apply(x, self.abs, self.delta,rexp_diff,fact),fact

