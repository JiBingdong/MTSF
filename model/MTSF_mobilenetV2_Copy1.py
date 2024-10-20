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
from .mobilenetV2_Copy1 import mobilenet_v2
from .coordatt import CA_Block_Y
import os
from torch import optim
from torch.utils import model_zoo
from .Decoders_Mobile_K3 import Decoder1,Decoder2,Decoder3,Decoder4

from torch.nn.modules.upsampling import Upsample
from torch.nn.functional import interpolate


class Upsample(nn.Module):
    def __init__(self, size, mode, align_corners=True):
        super(Upsample, self).__init__()
        self.interp = interpolate
        self.size = size
        self.mode = mode
        self.align_corners=align_corners

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode)
        return x

class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()
        self.conv4 = nn.Conv2d(1280,96,1)
        self.bach4 = nn.BatchNorm2d(96)
        self.relu4 = nn.GELU()
        self.up4 = Upsample(size=(14,14),mode="bilinear")
        self.conv3 = nn.Conv2d(96, 32, 1)
        self.bach3 = nn.BatchNorm2d(32)
        self.relu3 = nn.GELU()
        self.up3 = Upsample(size=(28, 28), mode="bilinear")
        self.conv2 = nn.Conv2d(32, 24, 1)
        self.bach2 = nn.BatchNorm2d(24)
        self.relu2 = nn.GELU()
        self.up2 = Upsample(size=(56, 56), mode="bilinear")
    def forward(self,hx1,hx2,hx3,hx4):

        hx3_ = hx3 + self.up4(self.relu4(self.bach4(self.conv4(hx4))))
        hx2_ = hx2 + self.up3(self.relu3(self.bach3(self.conv3(hx3_))))
        hx1_ = hx1 + self.up2(self.relu2(self.bach2(self.conv2(hx2_))))

        return hx1_, hx2_, hx3_, hx4


##### MTSM ####
class MTSM(nn.Module):

    def __init__(self,in_ch=3,out_ch=1, pretrained=False):
        super(MTSM,self).__init__()

        self.backbone = nn.Sequential(
            mobilenet_v2(pretrained))
        #self.FPN = FPN()

        self.stage4 = Decoder4()

        self.stage3 = Decoder3()

        self.stage2 = Decoder2()

        self.stage1 = Decoder1() #56

        self.outconv = nn.Conv2d(4 * out_ch, out_ch, kernel_size=1)

    def forward(self,x):
        hw_size = int(x.shape[3])
        hw_size_1 = int(hw_size/4)
        hw_size_2 = int(hw_size / 8)
        hw_size_3 = int(hw_size / 16)
        hw_size_4 = int(hw_size / 32)
        y_1 = torch.zeros(x.shape[0], 24, 16, hw_size_1, hw_size_1).cuda()
        y_2 = torch.zeros(x.shape[0], 32, 16, hw_size_2, hw_size_2).cuda()
        y_3 = torch.zeros(x.shape[0], 96, 16, hw_size_3, hw_size_3).cuda()
        y_4 = torch.zeros(x.shape[0], 1280, 16, hw_size_4, hw_size_4).cuda()
        for i in range(16):
            #x_ = torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[4])
            x_ = x[:, :, i, :, :]
            hx1, hx2, hx3, hx4 = self.backbone[0](x_)
            #hx1, hx2, hx3, hx4 = self.FPN(hx1,hx2,hx3,hx4)

            y_1[:, :, i, :, :] = hx1
            y_2[:, :, i, :, :] = hx2
            y_3[:, :, i, :, :] = hx3
            y_4[:, :, i, :, :] = hx4

        #hx = torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[4])
        #hx = x[:, :, 0, :, :]  # 只有预测的那张图片

        #hx1,hx2,hx3,hx4 = self.backbone[0](x)

        d1 = self.stage1(y_1)


        d2 = self.stage2(y_2)


        d3 = self.stage3(y_3)


        d4 = self.stage4(y_4)

        
        d0 = self.outconv(torch.cat(( d1, d2, d3, d4), 1))

        return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3),torch.sigmoid(d4)

