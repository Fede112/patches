import torch
import torch.nn as nn
import torch.nn.functional as F

# define the NN architecture
class Basic_CAE(nn.Module):
    def __init__(self):
        super(Basic_CAE, self).__init__()
        
        ## encoder layers ##
        self.conv1 = nn.Conv2d(1, 32, 3, padding = 1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding = 1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding = 1)
        self.pool = nn.MaxPool2d(2, 2)
        
        ## decoder layers ##
        self.t_conv1 = nn.Conv2d(128, 128, 3, padding = 1)
        self.t_conv2 = nn.Conv2d(128, 64, 3, padding = 1)
        self.t_conv3 = nn.Conv2d(64, 1, 3, padding = 1)
        self.upsampling = nn.modules.upsampling.Upsample(scale_factor=2, mode='nearest')
#         self.t_conv3 = nn.ConvTranspose2d(4, 2, 2, stride=2)
#         self.t_conv4 = nn.ConvTranspose2d(2, 1, 2, stride=2)
#         self.t_conv4 = nn.ConvTranspose2d(2, 1, 2, stride=2)
        
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
#         self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)

        
    def forward(self, x):
        ## encode ##
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))


       
        ## decode ##
        x = F.relu(self.t_conv1(x))
        x = self.upsampling(x)
        x = F.relu(self.t_conv2(x))
        x = self.upsampling(x)
        x = torch.sigmoid(self.t_conv3(x))
                
        return x


class Average_CAE(nn.Module):
    def __init__(self):
        super(Basic_CAE, self).__init__()
        
        ## encoder layers ##
        self.conv1 = nn.Conv2d(1, 32, 3, padding = 1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding = 1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding = 1)
        self.pool = nn.MaxPool2d(2, 2)
        
        ## decoder layers ##
        self.t_conv1 = nn.Conv2d(128, 128, 3, padding = 1)
        self.t_conv2 = nn.Conv2d(128, 64, 3, padding = 1)
        self.t_conv3 = nn.Conv2d(64, 1, 3, padding = 1)
        self.upsampling = nn.modules.upsampling.Upsample(scale_factor=2, mode='nearest')
#         self.t_conv3 = nn.ConvTranspose2d(4, 2, 2, stride=2)
#         self.t_conv4 = nn.ConvTranspose2d(2, 1, 2, stride=2)
#         self.t_conv4 = nn.ConvTranspose2d(2, 1, 2, stride=2)
        
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
#         self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)

        
    def forward(self, x):
        ## encode ##
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))


       
        ## decode ##
        x = F.relu(self.t_conv1(x))
        x = self.upsampling(x)
        x = F.relu(self.t_conv2(x))
        x = self.upsampling(x)
        x = torch.sigmoid(self.t_conv3(x))
                
        return x