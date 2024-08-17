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
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding = 1), #size = (28-5+2)/1 = 25
            nn.BatchNorm2d(1),
            nn.ReLU()
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
# for data in traindata:
#     # print(data[0])
#     data[0]
#     cnt+=1
#     if(cnt==10):
#         break