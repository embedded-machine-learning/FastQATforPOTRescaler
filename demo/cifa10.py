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


# from model.layer import *

# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.start = Start(8)
#         self.stop = Stop()
#         self.conv1 = BlockQuantN(3, 6, 5,2,outQuantDyn=True)
#         #self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = BlockQuantN(6, 16, 5,2,outQuantDyn=True)
#         self.conv3 = BlockQuantN(16, 32, 5,2)
#         self.bias = Bias(32)
#         self.fc1 = nn.Linear(512, 120)
#         self.fc3 = nn.Linear(120, 10)

#     def forward(self, x):
#         x = self.start(x)
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.bias(x)
#         x = self.stop(x)
#         x = torch.flatten(x, 1) # flatten all dimensions except batch
#         x = F.relu(self.fc1(x))
#         x = self.fc3(x)
#         return x

import torchvision

import model.resnet

net = model.resnet.resnet18(num_classes=10)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = net.to(device)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
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