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

class FMNIST_CNN(nn.Module):
    def __init__(self,
                 in_channels=1,
                 input_shape=[28,28],
                 code_length=64, 
                 classes = 10,
                 ):
        super(FMNIST_CNN,self).__init__()
        self.in_channels = in_channels
        self.input_shape = input_shape
        self.code_length = code_length
        self.classes = classes
       # Convolutonal Layer 1 
        self.convlayer1 = nn.Sequential(
            nn.Conv2d(in_channels,32, kernel_size = 3, padding = 1, bias=False),
            nn.PReLU(),
            nn.BatchNorm2d(32),
            )
        
        # Convolutonal Layer 2 
        self.convlayer2 = nn.Sequential(
            nn.Conv2d(32,32, kernel_size = 3,padding = 1),
            nn.PReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            )

        # Convolutonal Layer 3 
        self.convlayer3 = nn.Sequential(
            nn.Conv2d(32,64, kernel_size = 3,padding = 1),
            nn.PReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            )
        
        # Dense Layer 1
        self.fc1 = nn.Linear(64*7*7, self.code_length)
        
        # Dense Layer 2
        self.fc2 = nn.Linear(self.code_length, self.classes)
        
    
    def forward(self,x): #x = [batch,time,freq]
        x = self.convlayer1(x)
        x = self.convlayer2(x)
        x = self.convlayer3(x)
        x = torch.flatten(x,start_dim=1)
        z = F.relu(self.fc1(x))
        q = self.fc2(z)
        return q

    
class CIFAR10_CNN(torch.nn.Module):
    """Model Used by the paper introducing FedAvg"""

    def __init__(self):
        super(CIFAR10_CNN, self).__init__()
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