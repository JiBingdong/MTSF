"""
ResNet code gently borrowed from
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""
from __future__ import print_function, division, absolute_import
from collections import OrderedDict
import math
from torchvision.models.utils import load_state_dict_from_url
from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
from .S3D import s3d
from .Resnet18 import resnet18,resnet50
from .coordatt import CA_Block_Y
import os
from torch import optim
from torch.utils import model_zoo
from .Decoders_MTSF_Copy1 import Decoder5,Decoder4,Decoder3,Decoder2

#os.path.join('/home/jbd/allpython/forlunwen/u2net/code/model/')
class REBNCONV(nn.Module):
    def __init__(self,in_ch=3,out_ch=3,kernel_size=1,stride=1, padding=0):
        super(REBNCONV,self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch,out_ch,kernel_size,stride, padding)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.gelu_s1 = nn.GELU()

    def forward(self,x):

        hx = x
        xout = self.gelu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout
class DWCONV(nn.Module):
    def __init__(self,in_ch=3,out_ch=3):
        super(DWCONV,self).__init__()

        self.se = SEModule(in_ch,reduction=16)
        self.conv_s1 = nn.Conv2d(in_ch,in_ch,kernel_size=3, stride=1, padding=1, groups=in_ch, bias=False)
        self.bn_s1 = nn.BatchNorm2d(in_ch)
        self.gelu_s1 = nn.GELU()
        self.conv_s2 = nn.Conv2d(in_ch,out_ch,kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_s2 = nn.BatchNorm2d(out_ch)
        self.gelu_s2 = nn.GELU()

    def forward(self,x):

        hx = self.se(x)
        xout = self.gelu_s1(self.bn_s1(self.conv_s1(hx)))
        xout = self.gelu_s2(self.bn_s2(self.conv_s2(xout)))

        return xout

class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.gelu(x)
        return x
## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src,tar):

    #src = F.upsample(src,size=tar.shape[2:],mode='bilinear')
    src = F.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=False)

    return src
def _upsample_like_3D (src,tar):
    src = F.interpolate(src, size=tar.shape[2:], mode='trilinear', align_corners=True)
    return src
class DownConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=(), stride=1, padding=0):
        super(DownConv3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.gelu(x)
        return x
class Conv2d_1(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Conv2d_1, self).__init__()
        self.SEconv = SEModule(in_planes, reduction=16)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.SEconv(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.gelu(x)
        return x
class BasicConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.gelu(x)
        return x
##### MTSM( is our MTSF) ####
class MTSM(nn.Module):

    def __init__(self,in_ch=3,out_ch=1, pretrained=False):
        super(MTSM,self).__init__()
        #self.stage_backbone2D = resnet18(pretrained)
        #3DConv
        #self.stage_backbone3D = s3d(pretrained)
        '''
        self.backbone = nn.Sequential(
            resnet18(pretrained),
            s3d(pretrained))
        '''
        self.backbone = nn.Sequential(
            resnet50(pretrained),
            s3d(pretrained))
        self.downsample = nn.Sequential(
            BasicConv3d(192, 192, kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0)),
            BasicConv3d(192, 100, kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0)),
            BasicConv3d(100, 64, kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0)),
            BasicConv3d(480, 480, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),
            BasicConv3d(480, 240, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),
            BasicConv3d(240, 128, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),
            BasicConv3d(832, 400, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),
            BasicConv3d(400, 256, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),
            BasicConv3d(1024, 512, kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0)))
        self.Channel_2D = nn.Sequential(
            Conv2d_1(256, 64),
            Conv2d_1(512, 128),
            Conv2d_1(1024, 256),
            Conv2d_1(2048, 512))

        self.stage5_1 = Decoder5()

        self.stage4_1 = Decoder4()

        self.stage3_1 = Decoder3()

        self.stage2_1 = Decoder2()

        #kernel_size for 1 to 41
        #self.outconv = nn.Conv2d(4 * out_ch, out_ch, kernel_size=41, stride=1, padding=0, bias=False)
        self.outconv = nn.Conv2d(4 * out_ch, out_ch, kernel_size=1)

    def forward(self,x):

        #hx = x
        hx = torch.randn(x.shape[0],x.shape[1],x.shape[3],x.shape[4])
        hx = x[:,:,0,:,:] #只有预测的那张图片
        #stdge 3D
        y_1,y_2,y_3,y_4 = self.backbone[1](x)
       # print(y_1.size(),y_2.size(),y_3.size(),y_4.size())
        y_1 = self.downsample[0:3](y_1)
        y_1 = y_1.squeeze(2)
        # print(y_1.size())  #torch.Size([5, 256, 56, 56])
        y_2 = self.downsample[3:6](y_2)
        y_2 = y_2.squeeze(2)
        # print(y_2.size())  #torch.Size([5, 512, 28, 28])
        y_3 = self.downsample[6:8](y_3)
        y_3 = y_3.squeeze(2)
        # print(y_3.size())  #torch.Size([5, 1024, 14, 14])
        y_4 = self.downsample[8](y_4)
        y_4 = y_4.squeeze(2)
        # print(y_4.size())  #torch.Size([5, 1024, 7, 7])
        #stage 2D
        hx1,hx2,hx3,hx4 = self.backbone[0](hx)

        # downchannels for resnet50
        hx1 = self.Channel_2D[0](hx1)
        hx2 = self.Channel_2D[1](hx2)
        hx3 = self.Channel_2D[2](hx3)
        hx4 = self.Channel_2D[3](hx4)

        #print(hx1.size(),hx2.size(),hx3.size(),hx4.size())
        #hx1 = self.Channel_2D[0](hx1)
        d2 = self.stage2_1(torch.cat((y_1,hx1),1))

        #hx2 = self.Channel_2D[1](hx2)
        d3 = self.stage3_1(torch.cat((y_2,hx2),1))

        #hx3 = self.Channel_2D[2](hx3)
        d4 = self.stage4_1(torch.cat((y_3, hx3), 1))

        #hx4 = self.Channel_2D[3](hx4)
        d5 = self.stage5_1(torch.cat((y_4, hx4), 1))
        #print(d2.size(),d3.size(),d4.size(),d5.size())
        
        d1 = self.outconv(torch.cat(( d2, d3, d4, d5), 1))
        #d1 = _upsample_like(d1,d2)
        #return d0, d1, d2, d3, d4, d5
        return  torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(d4),torch.sigmoid(d5)

