import torch
import torchvision
import torchvision.transforms as transforms



transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 80

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
from model.activations import ReLU,PACT
from model.linear import Linear
from model.wrapped import FlattenM,MaxPool2d,Dropout
from model.blocks import ConvBnA



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.start = Start((1,3,1,1),8)
        self.stop = Stop((1,10))

        self.seq = nn.Sequential(
            ConvBnA(3,32,3,1,1,activation=PACT),
            ConvBnA(32,32,3,1,1,activation=PACT),
            MaxPool2d(2,2),
            Dropout(0.2),
            ConvBnA(32,64,3,1,1,activation=PACT),
            ConvBnA(64,64,3,1,1,activation=PACT),
            MaxPool2d(2,2),
            Dropout(0.3),
            ConvBnA(64,128,3,1,1,activation=PACT),
            ConvBnA(128,128,3,1,1,activation=PACT),
            MaxPool2d(2,2),
            Dropout(0.4),
            FlattenM(1),
            Linear(128*4*4,128,out_quant=PACT),
            Dropout(0.5),
            Linear(128,10)
        )

    def forward(self, x):
        x = self.start(x)
        x = self.seq(x)
        x = self.stop(x)
        return x

def eval(pr=True):
    global testloader,net,device
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
    net.train()
    if pr:
        print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    return 100 * correct / total

# net = model.resnet.resnet18(num_classes=10)
net = Net()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = net.to(device)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#optimizer = optim.Adam(net.parameters(),lr=0.001)

augm = nn.Sequential(torchvision.transforms.RandomResizedCrop(32,scale=(0.8,1.2)),torchvision.transforms.RandomHorizontalFlip()).to(device)

for epoch in range(100):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = augm(inputs.to(device))

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # if i == 0:    # print every 2000 mini-batches
    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 80:>2.3f} Test Acc:{eval(False):2.1f}%')
    running_loss = 0.0
            

print('Finished Training')


import os
if not os.path.exists("./demo/cifa10/"):
    os.mkdir("./demo/cifa10/")

torch.save(net.state_dict(),"./demo/cifa10/ckp.pt")