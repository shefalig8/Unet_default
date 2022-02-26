# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 20:35:50 2022

@author: Shefali Garg
"""

import torch
import torch.nn as nn


import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import numpy as np
import torchvision.models as models


def double_conv(in_c, out_c):
    conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3),
            nn.ReLU(inplace=True)
    )
    return conv

# skip connection, center crop
def crop_img_old(tensor, target_tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2  
    return tensor[:,:, delta:tensor_size - delta, delta:tensor_size - delta]



# skip connection, center crop
# Add bounds check
def crop_img(tensor, target_tensor):
    tw, th = target_tensor.size()[2],target_tensor.size()[3] 
    InW, InH = tensor.size()[2],tensor.size()[3] 
    deltaW = InW - tw
    deltaW = deltaW // 2  

    deltaH = InH - th
    deltaH = deltaH // 2  

    # NCHW
    return tensor[:,:, deltaW:(tw+deltaW),  deltaH:(th + deltaH)]


#  mobile-unet

# Default 
# Mobile-Unet
# Efficient-UNet
# 
class UNet(nn.Module):
    def __init__(self, in_channels=1, nclases=2, use_pretrained_wts=False):
        super(UNet, self).__init__()
        self.use_pretrained_wts = use_pretrained_wts
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv(in_channels,64)
        self.down_conv_2 = double_conv(64,128)
        self.down_conv_3 = double_conv(128,256)
        self.down_conv_4 = double_conv(256,512)
        self.down_conv_5 = double_conv(512,1024)
        
        self.up_trans_1 = nn.ConvTranspose2d(1024, 512 , kernel_size=2, stride=2 )
        self.up_conv_1 = double_conv(1024, 512)
        self.up_trans_2 = nn.ConvTranspose2d( 512, 256 , kernel_size=2, stride=2 )
        self.up_conv_2 = double_conv(512, 256)
        self.up_trans_3 = nn.ConvTranspose2d( 256, 128 , kernel_size=2, stride=2 )
        self.up_conv_3 = double_conv(256, 128)
        self.up_trans_4 = nn.ConvTranspose2d( 128, 64 , kernel_size=2, stride=2 )
        self.up_conv_4 = double_conv(128, 64)

        # self.up_trans_5 = nn.ConvTranspose2d( 64, 64 , kernel_size=4, stride=2 )
        # our case
        self.out = nn.Conv2d(64, nclases, kernel_size=1)

        # self.pretrained = VGG(use_pretrain=T)
        
    # w   300
    # " Default (Random Weihts)"
    def forward(self, image: th.Tensor):
        #encoder
        if self.use_pretrained_wts:
            pass
        else:
            x1 = self.down_conv_1(image)
            # print(' x1',x1.size())
            x2 = self.max_pool_2x2(x1)
            # print(' x2',x2.size())
            x3 = self.down_conv_2(x2)
            # print(' x3',x3.size())
            x4 = self.max_pool_2x2(x3)
            # print(' x4',x4.size())
            x5 = self.down_conv_3(x4)
            # print(' x5',x5.size())
            x6 = self.max_pool_2x2(x5)
            # print(' x6',x6.size())
            x7 = self.down_conv_4(x6)
            # print(' x7',x7.size())
            x8 = self.max_pool_2x2(x7)
            # print(' x8',x8.size())
            x9 = self.down_conv_5(x8)
            # print(' x9',x9.size())
            # print(x9.size())
            
            #decoder
            x= self.up_trans_1(x9)   
            # print('up x9',x.size())      /   
            y= crop_img(x7,x)  # input, out_size
            # print('x9 + x7 ',x7.size())
            x= self.up_conv_1(torch.cat([x,y],1))
            # print('x = x9+ x7 ',x.size())

            
            x= self.up_trans_2(x)
            # print('up x9',x.size())  
            y= crop_img(x5,x)

            x= self.up_conv_2(torch.cat([x,y],1))
            
            x= self.up_trans_3(x)
            y= crop_img(x3,x)
            x= self.up_conv_3(torch.cat([x,y],1))
            
            x= self.up_trans_4(x)
            y= crop_img(x1,x)
            x= self.up_conv_4(torch.cat([x,y],1))

            # x = self.up_trans_5(x)  ,

            x= self.out(x)

            # biliner interpolation
            x = torchvision.transforms.Resize([720, 960])(x)
            
            # x = 
            # x = nn.ConvTranspose2d( x.size[], 64 , kernel_size=2, stride=2 )(x)
        return x

# #  WXH  => mask 1d

# np.argmax( F.softmax(t) , d=1)  [1, W, H]
# 1120
import torchvision
if __name__ == "__main__":
    # 972, 1194
    # image = torch.rand((1,1,720,960))
    # 960/2 => 480
    image = torch.rand((1,3,920,1160))

    #print(image.shape)
    # img_resizer = torchvision.transforms.Resize([720, 960])

    # y = img_resizer(image)
    # print(y.size())
    # quit(0)
    # image = torch.rand((1,1,572,572))
    model = UNet(3,4)

    # move tensors to GPU if CUDA is available
    if train_on_gpu:
        model.cuda()

    # 572 = [388]   572/388  =

    #      572
    # 572  = (up 388) => 572 
    # # print(model)
    # # print(model(image))
    y = model.forward(image)
    print(y.shape)
    # mobile_net = models.mobilenet
    # print(mobile_net)
    # sdict = model.state_dict()

    # for k, v in sdict.items():
    #     print(k)
        
    # for param in model.parameters():
    #     # param.requires_grad = False
    #     print(' -> ',param.requires_grad)

    # # print(sdict['down_conv_1.0.bias'])

    # print(sdict['down_conv_1.0.bias'].shape)
        