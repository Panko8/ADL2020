import torch
from torch import nn
from torch import functional as F
import time
import numpy as np

class Image_model_by_distance(nn.Module):
    def __init__(self, in_channel=2):
        super(Image_model_by_distance, self).__init__()
        self.in_channel = in_channel
        self.pool = nn.MaxPool2d(2, stride=2) #no parameter
        #self.pools = nn.MaxPool2d(4, stride=4)
        self.lrelu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        self.drop2d = nn.Dropout2d(0, inplace=True)
        self.drop = nn.Dropout(0.5, inplace=True)
        
        self.conv1 = nn.Conv2d(in_channel, 16, (3,3), padding=1)
        self.norm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, (3,3), padding=1)
        self.norm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, (3,3), padding=1)
        self.norm3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, (3,3), padding=1)
        self.norm4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 255, (3,3), padding=1)
        self.norm5 = nn.BatchNorm2d(255)
        self.conv6 = nn.Conv2d(255*4, 255, (3,3), padding=1)
        self.norm6 = nn.BatchNorm2d(255)
        self.conv7 = nn.Conv2d(255, 255, (3,3), padding=1)
        self.norm7 = nn.BatchNorm2d(255)
        ##self.conv1x1 = nn.Conv2d(255, 64, (1,1) , padding=1)
        self.fc1 = nn.Linear(255*4*4, 128)
        self.fc2 = nn.Linear(128, 1)
    def forward(self, x, maps):
        '''
        x of shape (batch, in_channel, H=608, W=608)
        maps of shape (batch, 255*3=765, H=76',W=76')
        '''
        #extract little features
        batch_size = x.shape[0]
        x = self.pool(self.relu(self.norm1(self.drop2d(self.conv1(x)))))
        x = self.pool(self.relu(self.norm2(self.drop2d(self.conv2(x)))))
        x = self.pool(self.relu(self.norm3(self.drop2d(self.conv3(x)))))  # of shape(B.64,76,76)
        x = self.pool(self.relu(self.norm4(self.drop2d(self.conv4(x)))))  # of shape(B.128,38,38)
        x = self.pool(self.relu(self.norm5(self.drop2d(self.conv5(x)))))  # of shape(B,255,19,19)
        #concat feature maps
        x = torch.cat( (x,maps), dim=1 ) # of shape (B,255*4=1020,19,19)
        # shrink input
        x = self.pool(self.lrelu(self.norm6(self.drop2d(self.conv6(x))))) # of shape(B,255,9,9)
        x = self.pool(self.lrelu(self.norm7(self.drop2d(self.conv7(x))))) # of shape(B,255,4,4)
        ###x = self.conv1x1(x)
        #flatten
        x = x.view(batch_size, -1)
        x = self.lrelu(self.drop(self.fc1(x)))
        ###x = self.lrelu(self.drop(self.fc2(x)))
        x = self.lrelu(self.fc2(x))
        return x

def _debug():
    global model,p
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    in_channel=2
    concat_wh=19
    model=Image_model_by_distance(in_channel).to(device)
    x=torch.rand(3,in_channel,608,608).to(device)
    maps = torch.rand(3, 255*3, concat_wh, concat_wh).to(device)
    start=time.time()
    y=model(x, maps)
    duration=time.time()-start
    print(y)
    print(y.shape)
    print('Duration', duration)
    model_parameters = filter(lambda p: p[1].requires_grad, model.named_parameters())
    sparams = 0
    print("------ Parameters ------")
    for p in model_parameters:
        s=np.prod(p[1].size())
        print(p[0], s)
        sparams += s
    print('*Total Parameters', sparams)
    
    
    
if __name__ == "__main__":
    _debug()
