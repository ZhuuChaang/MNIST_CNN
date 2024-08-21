import torchvision.datasets as ds
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

from tqdm import tqdm

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq=nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding = 1), # (1,28,28) => (16,28,28)
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,padding=0), #(16,9,9)

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding = 1), #(32,9,9)
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,padding=0), #(32,3,3)

            nn.Flatten(),

            nn.Linear(32*3*3,128),
            nn.Linear(128,10),
        )
    
    def forward(self,x):
        output = self.seq(x)
        return output

datatrans = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize(mean=[0.5],std=[0.5])]
)


traindata = ds.MNIST(root="./dataset",train=True,transform=datatrans, download=False)
# testdata  = ds.MNIST(root="./dataset",train=False,download=False)

trainloader = DataLoader(traindata,batch_size=16,shuffle=True)
# testloader  = DataLoader(testdata, 128,shuffle=False)

ctrl=True #train or test

if(ctrl):
    classifier = CNN().to("cuda:0")

    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(list(classifier.parameters()),0.0001)


    for i in tqdm(range(20)):
        for data, label in trainloader:
            data=data.to("cuda:0")
            label=label.to("cuda:0")
            
            output = classifier(data)

            loss = criterion(output,label)
                        
            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"epoch: {i}, loss at {loss}")
    torch.cuda.empty_cache()

    torch.save(classifier,"models/classifier.pth")

else:
    pass