from PIL import Image
from os.path import join
import imageio
from torch import nn
from torch.nn.modules.linear import Linear
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
                    nn.Linear(784, 100),
                    nn.ReLU(), 
                    nn.Linear(100, 50),
                    nn.ReLU(), 
                    nn.Linear(50, 10),
                    nn.Sigmoid()
                    )
                                
        
    def forward(self, x):
        x = torch.flatten(x,start_dim = 1)
        z = self.fc(x)
        return z       
        

       
        
class FMNIST_CNN(nn.Module):
    def __init__(self,
                 in_channels=1,
                 input_shape=[28,28],
                 classes = 10,
                 drop_rate=0.2,
                 ):
        super(FMNIST_CNN,self).__init__()
        self.in_channels = in_channels
        self.input_shape = input_shape
        self.classes = classes
        self.drop_rate = drop_rate
       # Convolutonal Layer 1 
        self.convlayer1 = nn.Sequential(
            nn.Conv2d(in_channels,32, kernel_size = 5, padding = 0),
            nn.PReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            )
        
        # Convolutonal Layer 2 
        self.convlayer2 = nn.Sequential(
            nn.Conv2d(32,32, kernel_size = 5,padding = 0),
            nn.PReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(self.drop_rate, inplace=True)
            )

        
        # Dense Layer 1
        self.denselayer = nn.Sequential(
            nn.Linear(32*4*4, self.classes),
        )
        self.denselayer.name = 'fc'

        
    
    def forward(self,x): #x = [batch,time,freq]
        x = self.convlayer1(x)
        x = self.convlayer2(x)
        x = torch.flatten(x,start_dim=1)
        p = self.denselayer(x)
        return p


class CNN_CIFAR_dropout(torch.nn.Module):
    """Model Used by the paper introducing FedAvg"""

    def __init__(self):
        super(CNN_CIFAR_dropout, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=(3, 3)
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 3)
        )
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3)
        )

        self.fc1 = nn.Linear(4 * 4 * 64, 64)
        self.fc2 = nn.Linear(64, 10)

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout(x)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.dropout(x)
        x = x.view(-1, 4 * 4 * 64)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)
        return x
    
    
class CNN_CIFAR100_dropout(torch.nn.Module):
    """Model Used by the paper introducing FedAvg"""

    def __init__(self):
        super(CNN_CIFAR100_dropout, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=(3, 3)
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 3)
        )
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3)
        )

        self.fc1 = nn.Linear(4 * 4 * 64, 256)
        self.fc2 = nn.Linear(256, 100)

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout(x)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.dropout(x)
        x = x.view(-1, 4 * 4 * 64)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)
        return x
    
