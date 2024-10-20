import torch.nn as nn
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


class SepConv3d(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(SepConv3d, self).__init__()
        self.conv_s = nn.Conv3d(in_planes, out_planes, kernel_size=(1, 7, 7),
                                stride=(1, 1, 1), padding=(0, 3,3), bias=False)
        self.bn_s = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        #self.gelu_s = nn.ReLU()
        self.gelu_s = nn.GELU()
        # kernel_size = 3
        self.conv_t = nn.Conv3d(out_planes, out_planes, kernel_size=(7, 1, 1), stride=(1, 1, 1),
                                padding=(3, 0, 0), bias=False)
        self.bn_t = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.gelu_t = nn.GELU()
        #self.gelu_t = nn.ReLU()

    def forward(self, x):
        x = self.conv_s(x)
        x = self.bn_s(x)
        x = self.gelu_s(x)

        x = self.conv_t(x)
        x = self.bn_t(x)
        x = self.gelu_t(x)
        return x

class Decoder3D_block(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Decoder3D_block, self).__init__()
        self.conv1 = SepConv3d(in_planes, in_planes)
        self.conv2 = Conv3d_1(in_planes, in_planes*2)
        self.conv3 = Conv3d_1(in_planes*2, out_planes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class Conv3d_1(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Conv3d_1, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.gelu = nn.GELU()
        #self.gelu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.gelu(x)
        return x

class Conv3d_Down(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Conv3d_Down, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0), bias=False)
        self.bn = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.gelu = nn.GELU()
        #self.gelu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.gelu(x)
        return x

class Decoder4(nn.Module):
    def __init__(self, in_channel=1280, out_channel=1):
        super(Decoder4, self).__init__()

        #self.Downsample_0 = Conv3d_Down(in_channel, 96)  # 7*7,t=16
        self.Downsample_1 = Conv3d_Down(in_channel, 96) # 7*7,t=8
        self.deconvlayer1_6 = Decoder3D_block(in_planes=96, out_planes=96)  # 7*7,t=8
        self.upsample1_1 = Upsample(size=(8, 14, 14), mode='trilinear')

        #self.Downsample_1 = Conv3d_Down(96,96)  # 14*14,t=8
        self.deconvlayer1_5 = Decoder3D_block(in_planes=96, out_planes=32)  # 14*14,t=8
        self.upsample1_2 = Upsample(size=(8, 28, 28), mode='trilinear')

        self.Downsample_2 = Conv3d_Down(32, 32)  # 28*28,t=4
        self.deconvlayer1_4 = Decoder3D_block(in_planes=32, out_planes=24)  # 28*28,t=4
        self.upsample1_3 = Upsample(size=(4, 56, 56), mode='trilinear')

        self.deconvlayer1_3 = Decoder3D_block(in_planes=24, out_planes=12)  # 56*56,t=4
        self.upsample1_4 = Upsample(size=(4, 112, 112), mode='trilinear')

        self.Downsample_3 = Conv3d_Down(12, 12)  # 112*112,t=2
        self.deconvlayer1_2 = Decoder3D_block(12, 3)
        self.upsample1_5 = Upsample(size=(2, 224, 224), mode='trilinear')  # 224*224,t=2

        self.Downsample_4 = Conv3d_Down(3, 3)  # 224*224,t=1
        self.last_conv2 = nn.Conv2d(3, out_channel, kernel_size=1, stride=1, bias=True)  # 224*224,t=1

    def forward(self, x):
        x = self.Downsample_1(x)
        x = self.deconvlayer1_6(x)
        x = self.upsample1_1(x)


        x = self.deconvlayer1_5(x)
        x = self.upsample1_2(x)

        x = self.Downsample_2(x)
        x = self.deconvlayer1_4(x)
        x = self.upsample1_3(x)

        x = self.deconvlayer1_3(x)
        x = self.upsample1_4(x)

        x = self.Downsample_3(x)
        x = self.deconvlayer1_2(x)
        x = self.upsample1_5(x)

        x = self.Downsample_4(x)
        x = x.squeeze(2)
        x = self.last_conv2(x)

        return x



class Decoder3(nn.Module):
    def __init__(self, in_channel=96, out_channel=1):
        super(Decoder3, self).__init__()

        #self.Downsample_0 = Conv3d_Down(96, 96)  #14*14, t=16
        self.Downsample_1 = Conv3d_Down(96,96)   # 14*14,t=8
        self.deconvlayer1_5 = Decoder3D_block(in_planes=in_channel, out_planes=32)  # 14*14,t=8
        self.upsample1_1 = Upsample(size=(8, 28, 28), mode='trilinear')

        self.Downsample_2 = Conv3d_Down(32, 32)  # 28*28,t=4
        self.deconvlayer1_4 = Decoder3D_block(in_planes=32, out_planes=24)  # 28*28,t=4
        self.upsample1_2 = Upsample(size=(4, 56, 56), mode='trilinear')

        self.deconvlayer1_3 = Decoder3D_block(in_planes=24, out_planes=12)  # 56*56,t=4
        self.upsample1_3 = Upsample(size=(4, 112, 112), mode='trilinear')

        self.Downsample_3 = Conv3d_Down(12, 12)  # 112*112,t=2
        self.deconvlayer1_2 = Decoder3D_block(12, 3)
        self.upsample1_4 = Upsample(size=(2, 224, 224), mode='trilinear')  # 224*224,t=2

        self.Downsample_4 = Conv3d_Down(3, 3)  # 224*224,t=1
        self.last_conv2 = nn.Conv2d(3, out_channel, kernel_size=1, stride=1, bias=True)  # 224*224,t=1

    def forward(self, x):
        x = self.Downsample_1(x)
        x = self.deconvlayer1_5(x)
        x = self.upsample1_1(x)

        x = self.Downsample_2(x)
        x = self.deconvlayer1_4(x)
        x = self.upsample1_2(x)

        x = self.deconvlayer1_3(x)
        x = self.upsample1_3(x)

        x = self.Downsample_3(x)
        x = self.deconvlayer1_2(x)
        x = self.upsample1_4(x)

        x = self.Downsample_4(x)
        x = x.squeeze(2)
        x = self.last_conv2(x)

        return x


class Decoder2(nn.Module):
    def __init__(self, in_channel=32, out_channel=1):
        super(Decoder2, self).__init__()

        #self.Downsample_0 = Conv3d_Down(32, 32)  # 28*28,t=16
        self.Downsample_1 = Conv3d_Down(32, 32)  # 28*28,t=8
        self.Downsample_2 = Conv3d_Down(32, 32)  # 28*28,t=4
        self.deconvlayer1_4 = Decoder3D_block(in_planes=in_channel, out_planes=24)  # 28*28,t=4
        self.upsample1_1 = Upsample(size=(4,56,56), mode='trilinear')

        self.deconvlayer1_3 = Decoder3D_block(in_planes=24, out_planes=12)  # 56*56,t=4
        self.upsample1_2 = Upsample(size=(4,112,112), mode='trilinear')

        self.Downsample_3 = Conv3d_Down(12, 12)  # 112*112,t=2
        self.deconvlayer1_2 = Decoder3D_block(12, 3)
        self.upsample1_3 = Upsample(size=(2,224,224), mode='trilinear')  # 224*224,t=2

        self.Downsample_4 = Conv3d_Down(3, 3)  # 224*224,t=1
        self.last_conv2 = nn.Conv2d(3, out_channel, kernel_size=1, stride=1, bias=True)  # 224*224,t=1

    def forward(self, x):
        x = self.Downsample_2(self.Downsample_1(x))
        x = self.deconvlayer1_4(x)
        x = self.upsample1_1(x)

        x = self.deconvlayer1_3(x)
        x = self.upsample1_2(x)

        x = self.Downsample_3(x)
        x = self.deconvlayer1_2(x)
        x = self.upsample1_3(x)

        x = self.Downsample_4(x)
        x = x.squeeze(2)
        x = self.last_conv2(x)

        return x


class Decoder1(nn.Module):
    def __init__(self, in_channel=24, out_channel=1):
        super(Decoder1, self).__init__()

        #self.Downsample_0 = Conv3d_Down(24, 24)  # 56*56,t=16
        self.Downsample_1 = Conv3d_Down(24, 24)  # 56*56,t=8
        self.Downsample_2 = Conv3d_Down(24, 24)  # 56*56,t=4
        self.deconvlayer1_3 = Decoder3D_block(in_planes=in_channel, out_planes=12) #56*56,t=4
        self.upsample1_1 = Upsample(size=(4,112,112), mode='trilinear')

        self.Downsample_3 = Conv3d_Down(12, 12)  # 112*112,t=2
        self.deconvlayer1_2 = Decoder3D_block(12, 3)
        self.upsample1_2 = Upsample(size=(2,224,224), mode='trilinear') #224*224,t=2

        self.Downsample_4 = Conv3d_Down(3, 3)  # 224*224,t=1
        self.last_conv2 = nn.Conv2d(3, out_channel, kernel_size=1, stride=1, bias=True)  # 224*224,t=1


    def forward(self, x):

        x = self.Downsample_2(self.Downsample_1(x))
        x = self.deconvlayer1_3(x)
        x = self.upsample1_1(x)

        x = self.Downsample_3(x)
        x = self.deconvlayer1_2(x)
        x = self.upsample1_2(x)

        x = self.Downsample_4(x)
        x = x.squeeze(2)
        x = self.last_conv2(x)

        return x

