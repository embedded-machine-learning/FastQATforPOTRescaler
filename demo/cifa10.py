import torch
import torchvision
import torchvision.transforms as transforms



transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 200

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import os
import sys
sys.path.append(os.getcwd())

import torch.nn as nn
import torch.nn.functional as F


from model.Conversion import Start,Stop
from model.activations import ReLU
from model.batchnorm import BatchNorm2d
from model.convolution import Conv2d
from model.linear import Linear


from torch.nn.common_types import _size_2_t
from typing import Union

class BlockQuantN(nn.Module):
    """
    BlockQuantN A module with a Convolution BN and activation

    Per default the activation function is a leaky ReLu

    :param in_channels: Number of input channels
    :type in_channels: int
    :param out_channels: Number of output channels
    :type out_channels: int
    :param kernel_size: Kernel size for the Convolution
    :type kernel_size: _size_2_t
    :param stride: Stride for the Convolution, defaults to 1
    :type stride: _size_2_t, optional
    :param padding: padding for the Convolution, defaults to 0
    :type padding: Union[str, _size_2_t], optional
    :param dilation: Dilation for the Convolution, defaults to 1
    :type dilation: _size_2_t, optional
    :param groups: Groups for the Convolution, defaults to 1
    :type groups: int, optional
    :param padding_mode: Padding mode for the Convolution, defaults to "zeros"
    :type padding_mode: str, optional
    :param weight_quant: Overrides the default weight quantization for the Convolution, defaults to None
    :type weight_quant: _type_, optional
    :param weight_quant_bits: Number of bits for the Convolution Weight quantization, defaults to 8
    :type weight_quant_bits: int, optional
    :param weight_quant_channel_wise: If the Convolution Weight quantization should be done Layer-wise, defaults to False
    :type weight_quant_channel_wise: bool, optional
    :param weight_quant_args: Overrides the args for the Convolution Weight quantization, defaults to None
    :type weight_quant_args: _type_, optional
    :param weight_quant_kargs: Additional Named Arguments for the Convolution Weight quantization, defaults to {}
    :type weight_quant_kargs: dict, optional
    :param eps: EPS for the Batch-Norm , defaults to 1e-5
    :type eps: float, optional
    :param momentum: Momentum for the Batch-Norm, defaults to 0.1
    :type momentum: float, optional
    :param affine: Affine for the Batch-Norm, defaults to True
    :type affine: bool, optional
    :param track_running_stats: Trach running stats for the Batch-Norm, defaults to True
    :type track_running_stats: bool, optional
    :param fixed_n: If the batch-Norm should a single shift factor per layer, defaults to False
    :type fixed_n: bool, optional
    :param out_quant: Overrides the output quantization of the Batch-Norm, defaults to None
    :type out_quant: _type_, optional
    :param out_quant_bits: Number of bits for the output quantization of the Batch-Norm, defaults to 8
    :type out_quant_bits: int, optional
    :param out_quant_channel_wise: If the Batch-Norm output quantization should be done Channel-wise, defaults to False
    :type out_quant_channel_wise: bool, optional
    :param out_quant_args: Overrides the arguments for the batch-Norm output quantization, defaults to None
    :type out_quant_args: _type_, optional
    :param out_quant_kargs: Additional Named Arguments for the Batch-Norm output quantization, defaults to {}
    :type out_quant_kargs: dict, optional
    :param leaky_relu_slope: The LeakyRelu negative slope, defaults to 2**-6
    :type leaky_relu_slope: float, optional
    :param leaky_relu_inplace: If the Leaky Relu should be done inplace, defaults to False
    :type leaky_relu_inplace: bool, optional
    :param activation: Overrides the default activation function, e.g. nn.Sequential() for no activation, defaults to None
    :type activation: _type_, optional
    :param activation_args: Overrides the Arguments provided to the activation function, defaults to None
    :type activation_args: _type_, optional
    :param activation_kargs: Additional Named parameters for the activation function, defaults to {}
    :type activation_kargs: dict, optional
    """

    def __init__(
        self,
        # Convolution
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        padding_mode: str = "zeros",
        weight_quant=None,
        weight_quant_bits=8,
        weight_quant_channel_wise=False,
        weight_quant_args=None,
        weight_quant_kargs={},
        # Batch-Norm
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        fixed_n: bool = False,
        out_quant=None,
        out_quant_bits=8,
        out_quant_channel_wise=False,
        out_quant_args=None,
        out_quant_kargs={},
        # Activation
        leaky_relu_slope: float = 2**-6,
        leaky_relu_inplace: bool = False,
        activation=None,
        activation_args=None,
        activation_kargs={},
        # General stuff
        device=None,
        dtype=None,
    ) -> None:
        """
        Please see class documentation
        """
        super(BlockQuantN, self).__init__()
        self.conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
            weight_quant=weight_quant,
            weight_quant_bits=weight_quant_bits,
            weight_quant_channel_wise=weight_quant_channel_wise,
            weight_quant_args=weight_quant_args,
            weight_quant_kargs=weight_quant_kargs,
            out_quant=None,
            out_quant_bits=1,
            out_quant_channel_wise=False,
            out_quant_args=None,
            out_quant_kargs={},
        )
        self.bn = BatchNorm2d(
            num_features=out_channels,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            device=device,
            dtype=dtype,
            fixed_n=fixed_n,
            out_quant=out_quant,
            out_quant_bits=out_quant_bits,
            out_quant_channel_wise=out_quant_channel_wise,
            out_quant_args=out_quant_args,
            out_quant_kargs=out_quant_kargs,
        )


        self.activation = ReLU(8,(1,out_channels,1,1))

    def forward(self, invals) :

        fact = self.bn.get_weight_factor()

        x = self.conv(invals, fact)
        x = self.bn(x,self.activation)

        return x


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.start = Start((1,3,1,1),8)
        self.stop = Stop((1,32,1,1))
        self.conv1 = BlockQuantN(3 , 6 , 3,2,1)
        #self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = BlockQuantN(6 , 16, 3,2,1)
        self.conv3 = BlockQuantN(16, 32, 3,2,1)
        self.fc1 = nn.Linear(512, 120)
        self.fc3 = nn.Linear(120, 10)

    def forward(self, x):
        x = self.start(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.stop(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return x


# net = model.resnet.resnet18(num_classes=10)
net = Net()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = net.to(device)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
#optimizer = optim.Adam(net.parameters(),lr=0.001)



for epoch in range(20):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.to(device))
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 80 == 79:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 80:.3f}')
            running_loss = 0.0

print('Finished Training')

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
net.eval()
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        # calculate outputs by running images through the network
        outputs = net(images.to(device))
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

import os
if not os.path.exists("./demo/cifa10/"):
    os.mkdir("./demo/cifa10/")

torch.save(net.state_dict(),"./demo/cifa10/ckp.pt")