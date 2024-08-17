import torchvision.datasets as ds
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

class CNN(nn.modules):
    def __init__(self):
        super.__init__()
        self.seq=nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding = 1), 
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,padding=0),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding = 1), 
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,padding=0),

            nn.Linear(32*2*2,128),
            nn.Linear(128,10),
            nn.Softmax()
        )
    
    def forward(self,x):
        self.seq(x)


datatrans = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize(mean=[0.5],std=[0.5])]
)


traindata = ds.MNIST(root="./dataset",train=True,transform=datatrans, download=False)
# testdata  = ds.MNIST(root="./dataset",train=False,download=False)

trainloader = DataLoader(traindata,10,shuffle=True)
# testloader  = DataLoader(testdata, 128,shuffle=False)
# cnt=0
images, labels =  next(iter(traindata))

print(images.shape,labels)
