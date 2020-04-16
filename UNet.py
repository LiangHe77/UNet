#!/usr/bin/env python
# coding: utf-8

# In[2]:

import torch
import torch.nn as nn
from torchsummary import summary


# In[3]:


class BasicConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.conv(x)
        


# In[4]:


class BasicConv2(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,int(in_channels/2),kernel_size=3),
            nn.BatchNorm2d(int(in_channels/2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels/2),out_channels,kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.conv(x)


# In[5]:


class UpSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(mode='bilinear',scale_factor=2),
        )
    def forward(self,x):
        return self.upsample(x)


# In[12]:


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = BasicConv(1,64)
        self.conv2 = BasicConv(64,128)
        self.conv3 = BasicConv(128,256)
        self.conv4 = BasicConv(256,512)
        
        self.conv5 = BasicConv2(1024,256)
        self.conv6 = BasicConv2(512,128)
        self.conv7 = BasicConv2(256,64)
        
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.upsample1 = UpSample()
        self.upsample2 = UpSample()
        self.upsample3 = UpSample()
        self.upsample4 = UpSample()

        self.transition = nn.Sequential(
            nn.Conv2d(512,1024,kernel_size=3),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024,512,kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.end = nn.Sequential(
            nn.Conv2d(128,64,kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,2,kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True)
        )
        
    def forward(self,x):
        x_64_568 = self.conv1(x)
        x1_dim = x_64_568.shape[2]
        x_64_284 = self.pool(x_64_568)
        
        x_128_280 = self.conv2(x_64_284)
        x2_dim = x_128_280.shape[2]
        x_128_140 = self.pool(x_128_280)
        
        x_256_136 = self.conv3(x_128_140)
        x3_dim = x_256_136.shape[2]
        x_256_68 = self.pool(x_256_136)
        
        x_512_64 = self.conv4(x_256_68)
        x4_dim = x_512_64.shape[2]
        x_512_32 = self.pool(x_512_64)
        
        x_512_28 = self.transition(x_512_32)
        x_512_56 = self.upsample1(x_512_28)
        
        lower = int((x4_dim-x_512_56.shape[2])/2)
        upper = int(x4_dim-lower)
        x_1024_56 = torch.cat((x_512_56,(x_512_64)[:,:,lower:upper,lower:upper]),1)
        x_256_52 = self.conv5(x_1024_56)
        x_256_104 = self.upsample2(x_256_52)
        
        lower = int((x3_dim-x_256_104.shape[2])/2)
        upper = int(x3_dim-lower)
        x_512_104 = torch.cat((x_256_104,(x_256_136)[:,:,lower:upper,lower:upper]),1)
        x_128_100 = self.conv6(x_512_104)
        x_128_200 = self.upsample3(x_128_100)
        
        lower = int((x2_dim-x_128_200.shape[2])/2)
        upper = int(x2_dim-lower)
        x_256_200 = torch.cat((x_128_200,(x_128_280)[:,:,lower:upper,lower:upper]),1)
        x_64_196 = self.conv7(x_256_200)
        x_64_392 = self.upsample4(x_64_196)
        
        lower = int((x1_dim-x_64_392.shape[2])/2)
        upper = int(x1_dim-lower)
        x_128_392 = torch.cat((x_64_392,(x_64_568)[:,:,lower:upper,lower:upper]),1)
        out = self.end(x_128_392)
        return out


# In[11]:


if __name__== "__main__":
    input = torch.randn(1,1,572,572)
    net = UNet()
    output = net(input)
    print(output.shape)
