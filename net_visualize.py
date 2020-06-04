#import matplotlib.pyplot as plt
import numpy as np
import torch
#import torchvision
#import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import sys

'''
View by command:

tensorboard --logdir={LOGDIR}

'''


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.ones((2,16,4,4))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def demo():
    LOGDIR='tmp' #change this
    MODEL_NAME='trivial_name' #This name is for separting different models only, but all of them will be visualized!

    sample_net = Net()
    writer = SummaryWriter(LOGDIR+'/'+MODEL_NAME)
    #x=torch.randn((1,1,30,30))
    #y=torch.randn((1,1,30,30))
    x_y_stack = torch.randn((2,1,30,30)) # x of shape (channel, W, H) = (1,30,30)
    # p.s. (1,1,30,30) also works
    writer.add_graph(sample_net, x_y_stack)
    writer.close()
    
if __name__ == "__main__":
    print("Run demo...")
    demo()
    print("Created graph file within LOGDIR; please check!")

        
        

