import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

def linear(ni,no): return nn.Sequential(nn.Linear(ni,no),nn.ReLU())
def conv(ni,no,nf=3): return nn.Sequential(nn.Conv2d(ni,no,nf,padding=nf//2),nn.BatchNorm2d(no),nn.ReLU(),nn.MaxPool2d(2))
class Net(nn.Module):
    def __init__(self,nh=1280):
        super(Net, self).__init__()
        self.state_model = nn.Sequential(linear(2,128),
                                         linear(128,512),
                                         linear(512,512),
                                         linear(512,nh))
        self.image_model = EfficientNet.from_pretrained('efficientnet-b0')    
        self.image_model._fc= nn.Identity()
        self.fc = nn.Sequential(linear(nh,nh),nn.Linear(nh, 1))
    def forward(self, img, state):
        return self.fc(self.image_model(img)+self.state_model(state))