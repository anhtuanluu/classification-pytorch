import torch
import torch.nn as nn
from torchsummary import summary

class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.conv0 = nn.Conv2d(3, 32, 3)
        self.bn0 = nn.BatchNorm2d
        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu0 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(32, 64, 3)
        self.bn1 = nn.BatchNorm2d
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.bn2 = nn.BatchNorm2d
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.conv0(x)
        # x = self.bn0(x)
        x = self.relu0(x)
        x = self.pool0(x)

        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        # x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

def examplenet(**kwargs):
    net = Net(**kwargs)
    return net
