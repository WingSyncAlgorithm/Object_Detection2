import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from torchvision import datasets,  transforms
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from torchsummary import summary

class output:
    def __init__(self) -> None:
        pass
    def conv2d(self,k,H,W,Cout):
        return Cout,H-(k-1),W-(k-1)
    def maxpool(self,k,H,W,Cout):
        return Cout,int(H/k),int(W/k)
    
o = output
H,W = 128,128
C1,H1,W1 = o.conv2d(o,11,H,W,32)
print(C1,H1,W1)
C2,H2,W2 = o.maxpool(o,2,H1,W1,32)
print(C2,H2,W2)
C3,H3,W3 = o.conv2d(o,8,H2,W2,8)
print(C3,H3,W3)
C4,H4,W4 = o.maxpool(o,2,H3,W3,8)
print(C4,H4,W4)
    
class CNN_Model(nn.Module):
    # 列出需要哪些層
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.bn = nn.BatchNorm2d(3)
        # Convolution 1 , input_shape=(3,224,224)
        self.cnn1 = nn.Conv2d(3, 64, kernel_size=5, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        # Convolution 2
        self.cnn2 = nn.Conv2d(64, 8, kernel_size=11, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        # Fully connected 1 ,#input_shape=(8*50*50)
        self.fc = nn.Linear(C4 * H4 * W4, 3)
    # 列出forward的路徑，將init列出的層代入

    def forward(self, x):
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
model_1 = CNN_Model()
summary(model_1,(3,128,128))