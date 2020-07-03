import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torchvision
import torchvision.models as models
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
# %matplotlib inline
from numpy import moveaxis
from skimage import io, transform
from torch.autograd import Variable
from PIL import Image
import os
import glob
import pandas as pd
import random
import cv2
import plotly
# import bokeh
from sklearn.metrics import accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import warnings
warnings.filterwarnings("ignore")


def init_vgg():
    vgg = models.vgg16(pretrained=True)
    for param in vgg.parameters():
        param.requires_grad=False
    
    vgg.classifier[6] = torch.nn.Linear(4096,1024)
    vgg = nn.Sequential(vgg,
                        torch.nn.ReLU(),
                        torch.nn.Dropout(p=0.3),
                        torch.nn.Linear(1024, 128),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(p=0.2),
                        torch.nn.Linear(128, 10))
    return vgg

class Mononito(nn.Module):
    def __init__(self):
        super(Mononito, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
        self.relu1 = nn.ReLU()
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=3)
        self.dropout1 = nn.Dropout(p=0.3)
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.relu2 = nn.ReLU()
        
        self.maxpool2 = nn.MaxPool2d(kernel_size=3)
        self.dropout2 = nn.Dropout(p=0.3)
        
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.relu3 = nn.ReLU()
        
        self.maxpool3 = nn.MaxPool2d(kernel_size=3)
        self.dropout3 = nn.Dropout(p=0.3)
        
        self.fc1 = nn.Linear(1024, 192)
        self.dropout4 = nn.Dropout(p=0.3)
        
        self.fc2 = nn.Linear(192, 10)
#         self.softmax = torch.nn.Softmax(10)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        
        out = self.maxpool1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.relu2(out)
        
        out = self.maxpool2(out)
        out = self.dropout2(out)
        
        out = self.conv3(out)
        out = self.relu3(out)
        
        out = self.maxpool3(out)
        out = self.dropout3(out)
        
        out = out.view(out.size(0), -1)
        
        out = self.fc1(out)
        out = self.dropout4(out) 
        
        out = self.fc2(out)
#         output = self.log_softmax(x, dim=1)
        
        return out

