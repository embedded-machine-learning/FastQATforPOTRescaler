import os
import sys
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from model.Conversion import Start,Stop
from model.activations import ReLU,PACT
from model.linear import Linear
from model.wrapped import FlattenM,MaxPool2d,Dropout
from model.blocks import ConvBnA,ResidualBlock
from model.sequential import Sequential


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))])

batch_size = 80
epochs = 100


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






class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.start = Start(bits=8,size=(1,3,1,1),mode="auto",auto_runs=2)
        self.stop = Stop(size=(1,10))

        # self.seq = Sequential(
        #     ConvBnA(in_channels=3,out_channels=32,kernel_size=3,stride=2,padding=1,activation=PACT), 
        #     ResidualBlock(inplanes=32,planes=32),
        #     FlattenM(dim=1),
        #     Linear(in_features=16*16*32,out_features=10,weight_quant_channel_wise=True)
        # )

        self.seq = Sequential(
            ConvBnA(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=1,activation=PACT),
            # ConvBnA(32,32,3,1,1,activation=PACT),
            ResidualBlock(inplanes=32,planes=32),
            ResidualBlock(inplanes=32,planes=32),
            ResidualBlock(inplanes=32,planes=32),
            MaxPool2d(kernel_size=2,stride=2),
            Dropout(p=0.2),
            ConvBnA(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1,activation=PACT),
            # ConvBnA(64,64,3,1,1,activation=PACT),
            ResidualBlock(inplanes=64,planes=64),
            MaxPool2d(kernel_size=2,stride=2),
            Dropout(p=0.3),
            ConvBnA(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1,activation=PACT),
            # ConvBnA(128,128,3,1,1,activation=PACT),
            ResidualBlock(inplanes=128,planes=128),
            MaxPool2d(kernel_size=2,stride=2),
            Dropout(p=0.4),
            FlattenM(dim=1),
            Linear(in_features=128*4*4,out_features=128,out_quant=PACT,weight_quant_channel_wise=True),
            Dropout(p=0.5),
            Linear(in_features=128,out_features=10,weight_quant_channel_wise=True)
        )
        # self.seq = nn.Sequential(
        #     ConvBnA(3,32,3,1,1,activation=PACT),
        #     ConvBnA(32,32,3,1,1,activation=PACT),
        #     MaxPool2d(2,2),
        #     Dropout(0.2),
        #     ConvBnA(32,64,3,1,1,activation=PACT),
        #     ConvBnA(64,64,3,1,1,activation=PACT),
        #     MaxPool2d(2,2),
        #     Dropout(0.3),
        #     ConvBnA(64,128,3,1,1,activation=PACT),
        #     ConvBnA(128,128,3,1,1,activation=PACT),
        #     MaxPool2d(2,2),
        #     Dropout(0.4),
        #     FlattenM(1),
        #     Linear(128*4*4,128,out_quant=PACT),
        #     Dropout(0.5),
        #     Linear(128,10)
        # )

    def forward(self, x):
        x = self.start(x)
        x = self.seq(x)
        x = self.stop(x)
        return x
    
    def int_extract(self):
        seq = self.seq.int_extract()
        start = self.start.int_extract()
        stop = self.stop.int_extract()
        class Net_int(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.start = start
                self.stop = stop
                self.seq = seq
            def forward(self,x):
                x = self.start(x)
                x = self.seq(x)
                x = self.stop(x)
                return x
        return Net_int()

def eval(pr=True):
    global testloader,net,device
    global best
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
        print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:4.1f} %')
    if best < 100 * correct / total:
        best = 100 * correct / total

    return 100 * correct / total, best

# net = model.resnet.resnet18(num_classes=10)
net = Net()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = net.to(device)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9,weight_decay=1e-3)
sched= optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs,1e-5,verbose=True)
#optimizer = optim.Adam(net.parameters(),lr=0.001)

augm = nn.Sequential(torchvision.transforms.RandomResizedCrop(32,scale=(0.8,1.2)),torchvision.transforms.RandomHorizontalFlip()).to(device)

best = 0


if False:
    for epoch in range(epochs):  # loop over the dataset multiple times

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
        ev = eval(False)
        print(f'[{epoch + 1:3d}, {i + 1:5d}] loss: {running_loss / 80:6.3f} Test Acc:{ev[0]:3.1f}% Best test Acc:{ev[1]:3.1f}%')
        running_loss = 0.0
        torch.save(net.state_dict(),"./demo/cifa10/ckp.pt")
        sched.step()
                

    print('Finished Training')

else:
    net.load_state_dict(torch.load("./demo/cifa10/ckp.pt"))

print('Float')
eval()

from  model import __FLAGS__


device = 'cpu'
net = net.int_extract().to(device)
print('Int')
eval()
torch.save(net.state_dict(),"./demo/cifa10/Int.pt")

for param in net.parameters():
    param.requires_grad_(False)

__FLAGS__['ONNX_EXPORT'] = True

input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
output_names = [ "output1" ]
net.eval()
torch.onnx.export(net,         # model being run 
        torch.randn((1,3,32,32),requires_grad=False),       # model input (or a tuple for multiple inputs) 
        "./demo/cifa10/Int.onnx",       # where to save the model  
        export_params=True,  # store the trained parameter weights inside the model file 
        do_constant_folding=False,  # whether to execute constant folding for optimization 
        opset_version=10,
        input_names = ['modelInput'],   # the model's input names 
        output_names = ['modelOutput'], # the model's output names 
        dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                            'modelOutput' : {0 : 'batch_size'}}) 
print(" ") 
print('Model has been converted to ONNX') 

if not os.path.exists("./demo/cifa10/"):
    os.mkdir("./demo/cifa10/")

torch.save(net.state_dict(),"./demo/cifa10/final.pt")
