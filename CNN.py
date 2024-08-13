import torchvision.datasets as ds
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

traindata = ds.MNIST(root="./dataset",train=True, download=False)
testdata  = ds.MNIST(root="./dataset",train=False,download=False)

trainloader = DataLoader(traindata,128,shuffle=True)
testloader  = DataLoader(testdata, 128,shuffle=False)

