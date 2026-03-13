import torch
import torch.nn as nn

class Net(nn.Module):

    def __init__(self, in_channels=3, num_classes=10):
        super(Net,self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels,16,3),
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(16,16,3),
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.layer3 = nn.Sequential(
            nn.Conv2d(16,64,3),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer4 = nn.Sequential(
            nn.Conv2d(64,64,3),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer5 = nn.Sequential(
            nn.Conv2d(64,64,3),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer6 = nn.Sequential(
            nn.Conv2d(64,128,3),
            nn.BatchNorm2d(128),
            nn.ReLU())

        self.layer7 = nn.Sequential(
            nn.Conv2d(128,128,3),
            nn.BatchNorm2d(128),
            nn.ReLU())

        self.layer8 = nn.Sequential(
            nn.Conv2d(128,256,3),
            nn.BatchNorm2d(256),
            nn.ReLU())

        self.layer9 = nn.Sequential(
            nn.Conv2d(256,256,3),
            nn.BatchNorm2d(256),
            nn.ReLU())

        self.layer10 = nn.Sequential(
            nn.Conv2d(256,256,3),
            nn.BatchNorm2d(256),
            nn.ReLU())

        self.fc1 = nn.Sequential(
            nn.Linear(12*12*256,100),
            nn.ReLU(),
            nn.Dropout(0.5))

        self.fc2 = nn.Linear(100,num_classes)

    def forward(self,x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)

        x = x.view(x.size(0),-1)

        x = self.fc1(x)
        x = self.fc2(x)

        return x