import torch.nn as nn
from torch.nn.modules.upsampling import Upsample
from torch.nn.functional import interpolate
import time,torch


def position(H, W, is_cuda=True):
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc


def stride(x, stride):
    b, c, h, w = x.shape
    return x[:, :, ::stride, ::stride]


def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)


def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)


class ACmix(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_att=7, head=4, kernel_conv=3, stride=1, dilation=1):
        super(ACmix, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.head = head
        self.kernel_att = kernel_att
        self.kernel_conv = kernel_conv
        self.stride = stride
        self.dilation = dilation
        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))
        self.head_dim = self.out_planes // self.head

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv_p = nn.Conv2d(2, self.head_dim, kernel_size=1)

        self.padding_att = (self.dilation * (self.kernel_att - 1) + 1) // 2
        self.pad_att = torch.nn.ReflectionPad2d(self.padding_att)
        self.unfold = nn.Unfold(kernel_size=self.kernel_att, padding=0, stride=self.stride)
        self.softmax = torch.nn.Softmax(dim=1)

        self.fc = nn.Conv2d(3 * self.head, self.kernel_conv * self.kernel_conv, kernel_size=1, bias=False)
        self.dep_conv = nn.Conv2d(self.kernel_conv * self.kernel_conv * self.head_dim, out_planes,
                                  kernel_size=self.kernel_conv, bias=True, groups=self.head_dim, padding=1,
                                  stride=stride)

        self.reset_parameters()

    def reset_parameters(self):
        init_rate_half(self.rate1)
        init_rate_half(self.rate2)
        kernel = torch.zeros(self.kernel_conv * self.kernel_conv, self.kernel_conv, self.kernel_conv)
        for i in range(self.kernel_conv * self.kernel_conv):
            kernel[i, i // self.kernel_conv, i % self.kernel_conv] = 1.
        kernel = kernel.squeeze(0).repeat(self.out_planes, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=True)
        self.dep_conv.bias = init_rate_0(self.dep_conv.bias)

    def forward(self, x):
        q, k, v = self.conv1(x), self.conv2(x), self.conv3(x)
        scaling = float(self.head_dim) ** -0.5
        b, c, h, w = q.shape
        h_out, w_out = h // self.stride, w // self.stride

        # ### att
        # ## positional encoding
        pe = self.conv_p(position(h, w, x.is_cuda))

        q_att = q.view(b * self.head, self.head_dim, h, w) * scaling
        k_att = k.view(b * self.head, self.head_dim, h, w)
        v_att = v.view(b * self.head, self.head_dim, h, w)

        if self.stride > 1:
            q_att = stride(q_att, self.stride)
            q_pe = stride(pe, self.stride)
        else:
            q_pe = pe

        unfold_k = self.unfold(self.pad_att(k_att)).view(b * self.head, self.head_dim,
                                                         self.kernel_att * self.kernel_att, h_out,
                                                         w_out)  # b*head, head_dim, k_att^2, h_out, w_out
        unfold_rpe = self.unfold(self.pad_att(pe)).view(1, self.head_dim, self.kernel_att * self.kernel_att, h_out,
                                                        w_out)  # 1, head_dim, k_att^2, h_out, w_out

        att = (q_att.unsqueeze(2) * (unfold_k + q_pe.unsqueeze(2) - unfold_rpe)).sum(
            1)  # (b*head, head_dim, 1, h_out, w_out) * (b*head, head_dim, k_att^2, h_out, w_out) -> (b*head, k_att^2, h_out, w_out)
        att = self.softmax(att)

        out_att = self.unfold(self.pad_att(v_att)).view(b * self.head, self.head_dim, self.kernel_att * self.kernel_att,
                                                        h_out, w_out)
        out_att = (att.unsqueeze(1) * out_att).sum(2).view(b, self.out_planes, h_out, w_out)

        ## conv
        f_all = self.fc(torch.cat(
            [q.view(b, self.head, self.head_dim, h * w), k.view(b, self.head, self.head_dim, h * w),
             v.view(b, self.head, self.head_dim, h * w)], 1))
        f_conv = f_all.permute(0, 2, 1, 3).reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])

        out_conv = self.dep_conv(f_conv)

        return self.rate1 * out_att + self.rate2 * out_conv

class Upsample(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=False):
        super(Upsample, self).__init__()
        self.interp = interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners=align_corners

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode)
        return x
    
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
class DWCONV(nn.Module):
    def __init__(self,in_ch=3,out_ch=3):
        super(DWCONV,self).__init__()

        #self.se = SEModule(in_ch,reduction=16)
        self.conv_s1 = nn.Conv2d(in_ch,in_ch,kernel_size=7, stride=1, padding=3, groups=in_ch, bias=False)
        self.bn_s1 = nn.BatchNorm2d(in_ch)
        self.gelu_s1 = nn.GELU()
        self.conv_s2 = nn.Conv2d(in_ch,out_ch,kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_s2 = nn.BatchNorm2d(out_ch)
        self.gelu_s2 = nn.GELU()

    def forward(self,x):

        hx = x
        xout = self.gelu_s1(self.bn_s1(self.conv_s1(hx)))
        xout = self.gelu_s2(self.bn_s2(self.conv_s2(xout)))

        return xout
class bolck_ACmix(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(bolck_ACmix, self).__init__()
        
        
        self.conv1 = ACmix(in_planes, in_planes)
        self.conv2 = Conv2d_1(in_planes, in_planes*2)
        self.conv3 = Conv2d_1(in_planes*2, out_planes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x 

class bolck_DWCONV(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(bolck_DWCONV, self).__init__()
        self.conv1 = DWCONV(in_planes, in_planes)
        self.conv2 = Conv2d_1(in_planes, in_planes*2)
        self.conv3 = Conv2d_1(in_planes*2, out_planes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class Conv2d_1(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Conv2d_1, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.gelu(x)
        return x
class Conv2d_7(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Conv2d_1, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.gelu(x)
        return x

    
class Decoder5(nn.Module):
    def __init__(self, in_channel=512, out_channel=[512,256, 128, 64]):
        super(Decoder5, self).__init__()
        
        self.deconvlayer5_5 = bolck_ACmix(in_channel, 512)
        self.upsample5_5=Upsample(scale_factor=2, mode='bilinear')
        self.deconvlayer5_4 = bolck_DWCONV(out_channel[0], out_channel[1])
        self.upsample5_4=Upsample(scale_factor=2, mode='bilinear')
        self.deconvlayer5_3 = bolck_DWCONV(out_channel[1], out_channel[2])
        self.upsample5_3=Upsample(scale_factor=2, mode='bilinear')
        self.deconvlayer5_2 = bolck_DWCONV(out_channel[2], out_channel[3])  #56*56
        self.upsample5_2=Upsample(scale_factor=2, mode='bilinear')
        self.deconvlayer5_1 = bolck_DWCONV(64, 3)
        self.upsample5_1=Upsample(scale_factor=2, mode='bilinear')
        
        
        self.last_conv5=nn.Conv2d(3, 1, kernel_size=1, stride=1, bias=True)

    
    def forward(self, x):
        x=self.deconvlayer5_5(x)
        x = self.upsample5_5(x)
        x=self.deconvlayer5_4(x)
        x = self.upsample5_4(x)
        x=self.deconvlayer5_3(x)
        x = self.upsample5_3(x)
        x=self.deconvlayer5_2(x)
        x = self.upsample5_2(x)
        x=self.deconvlayer5_1(x)
        x = self.upsample5_1(x)
        x = self.last_conv5(x)
        
        return x
    
class Decoder4(nn.Module):
    def __init__(self, in_channel=256, out_channel=[256, 128, 64]):
        super(Decoder4, self).__init__()
        
        self.deconvlayer4_5 = bolck_ACmix(in_channel, out_channel[0])
        self.upsample4_5=Upsample(scale_factor=2, mode='bilinear')
        self.deconvlayer4_4 = bolck_DWCONV(out_channel[0], out_channel[1])
        self.upsample4_4=Upsample(scale_factor=2, mode='bilinear')
        self.deconvlayer4_3 = bolck_DWCONV(out_channel[1], out_channel[2])  #56*56
        self.upsample4_3=Upsample(scale_factor=2, mode='bilinear')
        self.deconvlayer4_2 = bolck_DWCONV(out_channel[2], 3)
        self.upsample4_2=Upsample(scale_factor=2, mode='bilinear')
        #self.deconvlayer4_1 = bolck_DWCONV(64, 3)
        #self.upsample4_1=Upsample(scale_factor=2, mode='bilinear')
        
        
        self.last_conv4=nn.Conv2d(3, 1, kernel_size=1, stride=1, bias=True)
        

    def forward(self, x):
       
        x =self.deconvlayer4_5(x)
        x = self.upsample4_5(x)
        x =self.deconvlayer4_4(x)
        x = self.upsample4_4(x)
        x =self.deconvlayer4_3(x)
        x = self.upsample4_3(x)
        x =self.deconvlayer4_2(x)
        x = self.upsample4_2(x)
        #x = self.deconvlayer4_1(x)
        #x = self.upsample4_1(x)
        x = self.last_conv4(x)

        return x

class Decoder3(nn.Module):
    def __init__(self, in_channel=128, out_channel=[128,64], out_sigmoid=False):
        super(Decoder3, self).__init__()
        
        self.out_sigmoid=out_sigmoid
        
        self.deconvlayer3_4 = bolck_ACmix(in_channel, out_channel[0])
        self.upsample3_4=Upsample(scale_factor=2, mode='bilinear')
        self.deconvlayer3_3 = bolck_DWCONV(out_channel[0], out_channel[1])  #56*56
        self.upsample3_3=Upsample(scale_factor=2, mode='bilinear')
        self.deconvlayer3_2 = bolck_DWCONV(out_channel[1], 3)
        self.upsample3_2=Upsample(scale_factor=2, mode='bilinear')
        #self.deconvlayer3_1 = bolck_DWCONV(out_channel[1], 3)
        #self.upsample3_1=Upsample(scale_factor=2, mode='bilinear')
        
        
        self.last_conv3=nn.Conv2d(3, 1, kernel_size=1, stride=1, bias=True)
    
    def forward(self, x):
       
        x=self.deconvlayer3_4(x)
        x = self.upsample3_4(x)
        x=self.deconvlayer3_3(x)
        x = self.upsample3_3(x)
        x=self.deconvlayer3_2(x)
        x = self.upsample3_2(x)
        #x= self.deconvlayer3_1(x)
        #x = self.upsample3_1(x)
    
        x = self.last_conv3(x)
            
        return x

class Decoder2(nn.Module):
    def __init__(self, in_channel=64, out_channel=64, out_sigmoid=False):
        super(Decoder2, self).__init__()
        
        self.out_sigmoid=out_sigmoid
        
        self.deconvlayer2_3 = bolck_ACmix(in_planes=in_channel, out_planes=64) #56*56
        self.upsample2_3=Upsample(scale_factor=2, mode='bilinear')
        self.deconvlayer2_2 = bolck_DWCONV(64, 3)
        self.upsample2_2=Upsample(scale_factor=2, mode='bilinear')
        #self.deconvlayer2_1 = bolck_DWCONV(64, 3)
        #self.upsample2_1=Upsample(scale_factor=2, mode='bilinear')
        
        self.last_conv2=nn.Conv2d(3, 1, kernel_size=1, stride=1, bias=True)

    def forward(self, x):
       
        x=self.deconvlayer2_3(x)
        x = self.upsample2_3(x)
        x=self.deconvlayer2_2(x)
        x = self.upsample2_2(x)
        #x= self.deconvlayer2_1(x)
        #x = self.upsample2_1(x)
        
        x = self.last_conv2(x)

        return x

