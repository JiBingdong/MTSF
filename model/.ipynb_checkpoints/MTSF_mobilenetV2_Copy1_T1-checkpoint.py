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

##### MTSM ####
class MTSM(nn.Module):

    def __init__(self,in_ch=3,out_ch=1, pretrained=False):
        super(MTSM,self).__init__()

        self.backbone = nn.Sequential(
            mobilenet_v2(pretrained))


        self.stage4 = Decoder4()

        self.stage3 = Decoder3()

        self.stage2 = Decoder2()

        self.stage1 = Decoder1() #56

        self.outconv = nn.Conv2d(4 * out_ch, out_ch, kernel_size=1)

    def forward(self,x,y_1,y_2,y_3,y_4):

        hx1, hx2, hx3, hx4 = self.backbone[0](x)
        y_1[:, :, 0, :, :] = hx1
        y_2[:, :, 0, :, :] = hx2
        y_3[:, :, 0, :, :] = hx3
        y_4[:, :, 0, :, :] = hx4

        #hx = torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[4])
        #hx = x[:, :, 0, :, :]  # 只有预测的那张图片

        #hx1,hx2,hx3,hx4 = self.backbone[0](x)

        d1 = self.stage1(y_1)


        d2 = self.stage2(y_2)


        d3 = self.stage3(y_3)


        d4 = self.stage4(y_4)

        
        d0 = self.outconv(torch.cat(( d1, d2, d3, d4), 1))

        return y_1, y_2, y_3, y_4, torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3),torch.sigmoid(d4)

