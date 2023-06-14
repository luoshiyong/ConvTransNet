from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from net.defconv import DefC
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, trunc_normal_
from functools import partial
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return DefC(in_planes, out_planes, kernel_size=3, stride=stride,padding=dilation, bias=False)

class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            conv3x3(in_channels,out_channels,dilation=dilation),
            # nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)
import numpy as np

class DSPP(nn.Module):
    def __init__(self, in_channels,out_ch=256, split=[8,4,2,1,1],atrous_rates=[6, 12, 18]):
        super(DSPP, self).__init__()
        self.allch = np.array(split).sum()
        out_channels = [idx*(in_channels//self.allch) for idx in split]
        self.out_ch = out_channels
        self.split = split
        modules = []

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(out_channels[0], out_channels[0], rate1))
        modules.append(ASPPConv(out_channels[1], out_channels[1], rate2))
        modules.append(ASPPConv(out_channels[2], out_channels[2], rate3))
        modules.append(ASPPPooling(out_channels[3], out_channels[3]))
        modules.append(nn.Sequential(
            nn.Conv2d(out_channels[4], out_channels[4], 1, bias=False),
            nn.BatchNorm2d(out_channels[4]),
            nn.ReLU(inplace=True)))
        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(in_channels*2, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1) )

    def forward(self, y):
        x = torch.split(y,self.out_ch,dim=1)
        res = []
        for idx in range(len(self.convs)):
            # print("x.shape = {}|conv = {}".format(x[idx].shape,self.convs[idx]))
            res.append(self.convs[idx](x[idx]))
        res = torch.cat(res, dim=1)
        return self.project(torch.cat([res,y],dim=1))

class U_Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, args):
        super(U_Net, self).__init__()
        self.args = args
        in_ch = 3
        out_ch = 2

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        # d1 = self.active(out)

        return out
class U_Net_DSPP(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, args):
        super(U_Net_DSPP, self).__init__()
        self.args = args
        in_ch = 3
        out_ch = 2

        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        self.dspp1 = DSPP(in_channels=256, out_ch=256)
        self.dspp2 = DSPP(in_channels=128, out_ch=128)
        self.dspp3 = DSPP(in_channels=64, out_ch=64)
    # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        e4 = self.dspp1(e4)  # [b, 256, 42,42]
        e3 = self.dspp2(e3)  # [b,128, 84,84]
        e2 = self.dspp3(e2)  # [b,64, 168, 168]

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        # d1 = self.active(out)

        return out
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = 12# num_heads
        head_dim = (dim+147)//12# dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, (dim+147) * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear((dim+147), dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, (C+147)// self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # print("q shape = ",q.shape)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # print("********************attention shape = ",attn.shape)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, (C+147))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # x = x.transpose(2,1)
        # print("x ***********shape = ",x.shape)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class ConvBlock(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1, res_conv=False, act_layer=nn.ReLU, groups=1,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), drop_block=None, drop_path=None):
        super(ConvBlock, self).__init__()

        expansion = 4
        med_planes = outplanes // expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=stride, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(med_planes, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(outplanes)
        self.act3 = act_layer(inplace=True)

        if res_conv:
            self.residual_conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, padding=0, bias=False)
            self.residual_bn = norm_layer(outplanes)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, x_t=None, return_x_2=True):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x) if x_t is None else self.conv2(x + x_t)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x += residual
        x = self.act3(x)

        if return_x_2:
            return x, x2
        else:
            return x
class FCUDown(nn.Module):
    """ CNN feature maps -> Transformer patch embeddings
    """

    def __init__(self, inplanes, outplanes, dw_stride, act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(FCUDown, self).__init__()
        self.dw_stride = dw_stride

        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.sample_pooling = nn.AvgPool2d(kernel_size=dw_stride, stride=dw_stride)

        self.ln = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, x_t):
        x = self.conv_project(x)  # [N, C, H, W]
        x = self.sample_pooling(x).flatten(2)
        # print("&&&&&&&& shape = ", x_t[:, 0][:, None, :].shape)
        x = self.ln(x)
        x = self.act(x)

        # x = torch.cat([x_t[:, 0][:, None, :], x], dim=1)

        return x


class FCUUp(nn.Module):
    """ Transformer patch embeddings -> CNN feature maps
    """

    def __init__(self, inplanes, outplanes, up_stride, act_layer=nn.ReLU,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6),):
        super(FCUUp, self).__init__()

        self.up_stride = up_stride
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.bn = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, H, W):
        B, _, C = x.shape
        # [N, 197, 384] -> [N, 196, 384] -> [N, 384, 196] -> [N, 384, 14, 14]
        x_r = x.transpose(1, 2).reshape(B, C, H, W)
        x_r = self.act(self.bn(self.conv_project(x_r)))

        return F.interpolate(x_r, size=(H * self.up_stride, W * self.up_stride))
class GlobalDown(nn.Module):
    """ CNN feature maps -> Transformer patch embeddings
    """

    def __init__(self, inplanes, outplanes, dw_stride, act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(GlobalDown, self).__init__()
        self.dw_stride = dw_stride

        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0) #
        self.sample_pooling = nn.AvgPool2d(kernel_size=dw_stride, stride=dw_stride)
        self.ln = norm_layer(outplanes)
        self.act = act_layer()
    def forward(self, x, x_t):
        # xglbao [2, 32, 441]    [2, 64, 441]     [2, 128, 441]
        x = self.conv_project(x)  # [N, C, H, W]
        x = self.sample_pooling(x).flatten(2)
        # print("x shape((((((((((((((((((  ",x.shape)
        x = self.ln(x)
        x = self.act(x)

        # x = torch.cat([x_t[:, 0][:, None, :], x], dim=1)

        return x
class LocalDown(nn.Module):
    """ CNN feature maps -> Transformer patch embeddings
    """

    def __init__(self, inplanes, outplanes, dw_stride, act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(LocalDown, self).__init__()
        self.dw_stride = dw_stride

        self.conv_project = nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, padding=0) #
        self.local_embedding = Rearrange('b c (h p1) (w p2) -> b (c h w) (p1 p2)', p1=21, p2=21)
        # self.squeeze_conv = nn.Conv2d(inplanes*dw_stride*dw_stride,inplanes,3,groups=dw_stride*dw_stride)
        self.ln = norm_layer(outplanes)
        self.act = act_layer()
        self.catnorm = norm_layer(441)
        self.ffn = nn.Linear(441,441)
    def forward(self, x, x_t):
        # xglbao [2, 32, 441]    [2, 64, 441]     [2, 128, 441]
        x = self.conv_project(x)  # [N, C, H, W]
        x = self.local_embedding(x)  #
        # x = self.catnorm(x)
        # x = self.ffn(x)
        # xlocla [2, 512, 441]   [2, 256, 441]    [2, 128, 441]
        # print("x local shape = ",x.shape)
        # x = self.squeeze_conv(x)
        x = self.ln(x)
        x = self.act(x)

        # x = torch.cat([x_t[:, 0][:, None, :], x], dim=1)

        return x

class MixDown(nn.Module):
    """ CNN feature maps -> Transformer patch embeddings
    """

    def __init__(self, inplanes, outplanes, dw_stride, act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(MixDown, self).__init__()
        self.dw_stride = dw_stride

        self.conv_project = nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, padding=0) #
        self.sample_pooling = nn.AvgPool2d(kernel_size=dw_stride, stride=dw_stride)
        self.local_embedding = Rearrange('b c (h p1) (w p2) -> b (c h w) (p1 p2)', p1=21, p2=21)
        # self.squeeze_conv = nn.Conv2d(inplanes*dw_stride*dw_stride,inplanes,3,groups=dw_stride*dw_stride)
        self.ln = norm_layer(outplanes)
        self.act = act_layer()
        self.catnorm = norm_layer(882)
        self.ffn = nn.Linear(882,441)
    def forward(self, x, x_t):
        # xglbao [2, 32, 441]    [2, 64, 441]     [2, 128, 441]
        xglobal = self.sample_pooling(x).flatten(2)
        xglobal = xglobal.repeat(1,self.dw_stride*self.dw_stride,1)
        x = self.conv_project(x)  # [N, C, H, W]
        x = self.local_embedding(x)  #
        x = torch.cat([x,xglobal],dim=-1)
        x = self.catnorm(x)
        x = self.ffn(x)
        # xlocla [2, 512, 441]   [2, 256, 441]    [2, 128, 441]
        # print("x local shape = ",x.shape)
        # x = self.squeeze_conv(x)
        x = self.ln(x)
        x = self.act(x)

        # x = torch.cat([x_t[:, 0][:, None, :], x], dim=1)

        return x


class MixUp(nn.Module):
    """ Transformer patch embeddings -> CNN feature maps
    """

    def __init__(self, inplanes, outplanes, up_stride, act_layer=nn.ReLU,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6),):
        super(MixUp, self).__init__()

        self.up_stride = up_stride
        self.res_out = Rearrange('b (c p1 p2) (w1 w2) -> b c (p1  w1) (p2 w2)',p1=up_stride,p2=up_stride,w1=21,w2=21)

        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.bn = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x):
        x_r = self.res_out(x)
        # print("x shape = {} |x_r shape = {}".format(x.shape,x_r.shape))
        x_r = self.act(self.bn(self.conv_project(x_r)))

        return x_r
class LocalUp(nn.Module):
    """ Transformer patch embeddings -> CNN feature maps
    """

    def __init__(self, inplanes, outplanes, up_stride, act_layer=nn.ReLU,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6),):
        super(LocalUp, self).__init__()

        self.up_stride = up_stride
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.bn = norm_layer(outplanes)
        self.act = act_layer()
        self.res_out = Rearrange('b (c p1 p2) (w1 w2) -> b c (p1  w1) (p2 w2)',p1=up_stride,p2=up_stride,w1=21,w2=21)

    def forward(self, x):
        # print("x shape = ",x.shape)  # [2, 512, 441])
        x = self.res_out(x)
        # print("x shape = ",x.shape)  # [2, 32, 84, 84]

        x_r = self.act(self.bn(self.conv_project(x)))

        return x_r

class GlobalUp(nn.Module):
    """ Transformer patch embeddings -> CNN feature maps
    """

    def __init__(self, inplanes, outplanes, up_stride, act_layer=nn.ReLU,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6),):
        super(GlobalUp, self).__init__()

        self.up_stride = up_stride
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.bn = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x):
        # print("x shape = ",x.shape)
        B, C, len = x.shape
        H,W = 21,21
        # [N, 197, 384] -> [N, 196, 384] -> [N, 384, 196] -> [N, 384, 14, 14]
        x_r = x.reshape(B, C ,H , W)
        x_r = self.act(self.bn(self.conv_project(x_r)))

        return F.interpolate(x_r, size=(H * self.up_stride, W * self.up_stride))
class ConvTransBlock(nn.Module):
    """
    Basic module for ConvTransformer, keep feature maps for CNN block and patch embeddings for transformer encoder block
    """

    def __init__(self, get_type,inplanes, outplanes, res_conv, stride, dw_stride, embed_dim, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., groups=1):

        super(ConvTransBlock, self).__init__()
        expansion = 4
        self.cnn_block = ConvBlock(inplanes=inplanes, outplanes=outplanes, res_conv=res_conv, stride=stride, groups=groups)
        self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, groups=groups)


        if  get_type=="global":
            self.squeeze_block = GlobalDown(inplanes=outplanes // expansion, outplanes=embed_dim, dw_stride=dw_stride)
            self.expand_block = GlobalUp(inplanes=embed_dim, outplanes=outplanes // expansion, up_stride=dw_stride)
        elif get_type=="local":
            self.squeeze_block = LocalDown(inplanes=outplanes // expansion, outplanes=embed_dim, dw_stride=dw_stride)
            self.expand_block = LocalUp(inplanes=outplanes // expansion, outplanes=outplanes // expansion, up_stride=dw_stride)
        elif get_type=="channel":
            self.squeeze_block = FCUDown(inplanes=outplanes // expansion, outplanes=embed_dim, dw_stride=dw_stride)
            self.expand_block = FCUUp(inplanes=embed_dim, outplanes=outplanes // expansion, up_stride=dw_stride)
        elif get_type=="mix":
            self.squeeze_block = MixDown(inplanes=outplanes // expansion, outplanes=embed_dim, dw_stride=dw_stride)
            self.expand_block = MixUp(inplanes=outplanes // expansion, outplanes=outplanes // expansion, up_stride=dw_stride)
        else:
            raise Exception('not implement!')
        self.trans_block = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate)

        self.dw_stride = dw_stride
        self.embed_dim = embed_dim

    def forward(self, x, x_t):
        # x[b,128,84,84]  x_t[b,441,768]
        x, x2 = self.cnn_block(x)
        # x[b,128,84,84]    x2[2,32,84,84]
        _, _, H, W = x2.shape
        # fcudown
        # print("x2 shape = ",x2.shape)
        x_st = self.squeeze_block(x2, x_t)
        # xst->[b,441,384]
        # attention
        # print("xt shape= {} | xst shape ={}".format(x_t.shape,x_st.shape))
        x_t = self.trans_block(x_st)

        # fcuup
        x_t_r = self.expand_block(x_t)
        # fusion
        x = self.fusion_block(x, x_t_r, return_x_2=False)
        # return x, x_t
        return x,x_t_r,x_t
class F3F(nn.Module):
    """
        Feature fusion module
    """

    def __init__(self, cl, ch, cx):
        super(F3F, self).__init__()

        self.l1x1 = nn.Conv2d(cl,cx,kernel_size=3,stride=1,padding=1)
        self.h1x1 = nn.Conv2d(ch, cx, kernel_size=1, stride=1, padding=0)
        #
        self.x1x1 = nn.Conv2d(cx, 1, kernel_size=1, stride=1, padding=0)
        self.xup = nn.Upsample(scale_factor=2)
        self.xdown = nn.Conv2d(cx,cx,kernel_size=2,stride=2,padding=0)
        #
        self.l2x2 = nn.Conv2d(cx, cx, kernel_size=2, stride=2, padding=0)
        self.hup = nn.Upsample(scale_factor=2)
        #
        self.outconv = nn.Conv2d(cx*3,cx,kernel_size=3,stride=1,padding=1)
    def forward(self, xl,xx,xh):
        xl = self.xup(xx) + self.l1x1(xl)
        xh = self.xdown(xx) + self.h1x1(xh)
        weight = F.sigmoid(self.x1x1(xx))
        xl = self.l2x2(xl)*weight
        xh = self.hup(xh)*weight
        out = self.outconv(torch.cat([xl,xx*weight,xh],dim=1))
        return xx+out

class U_Net_DSPP_Transformer(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, get_type,base_channel=64,embed_dim=441,num_heads=9,channel_ratio=2,patch_size=16,mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super(U_Net_DSPP_Transformer, self).__init__()
        in_chans = 3
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(3, 64)
        self.Conv2 = conv_block(64, 64)

        # self.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 1 / 2 [112, 112]
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU(inplace=True)
        # 1 stage
        self.trans_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 12)]
        stage_1_channel = int(base_channel * channel_ratio)
        trans_dw_stride = patch_size // 4
        self.conv_1 = ConvBlock(inplanes=64, outplanes=stage_1_channel, res_conv=True, stride=1)
        self.trans_patch_conv = nn.Conv2d(64, embed_dim, kernel_size=trans_dw_stride, stride=trans_dw_stride, padding=0)
        self.trans_1 = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.trans_dpr[0],
                             )
        self.first_stage = ConvTransBlock(get_type,
                        stage_1_channel, stage_1_channel, False, 1, dw_stride=trans_dw_stride, embed_dim=embed_dim,
                        num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate)
        self.first_stage2 = ConvTransBlock(get_type,
            stage_1_channel, stage_1_channel, False, 1, dw_stride=trans_dw_stride, embed_dim=embed_dim,
            num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate)
        self.second_conv = conv_block(128, 256)
        self.second_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.second_stage = ConvTransBlock(get_type,
            256, 256, False, 1, dw_stride=trans_dw_stride//2, embed_dim=embed_dim,
            num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate)
        self.second_stage2 = ConvTransBlock(get_type,
            256, 256, False, 1, dw_stride=trans_dw_stride // 2, embed_dim=embed_dim,
            num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate)
        self.second_stage3 = ConvTransBlock(get_type,
                                            256, 256, False, 1, dw_stride=trans_dw_stride // 2, embed_dim=embed_dim,
                                            num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                            qk_scale=qk_scale,
                                            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate)
        self.third_conv = conv_block(256, 512)
        self.third_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.third_stage = ConvTransBlock(get_type,
            512, 512, False, 1, dw_stride=trans_dw_stride // 4, embed_dim=embed_dim,
            num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate)
        self.third_stage2 = ConvTransBlock(get_type,
            512, 512, False, 1, dw_stride=trans_dw_stride // 4, embed_dim=embed_dim,
            num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate)
        self.third_stage3 = ConvTransBlock(get_type,
                                           512, 512, False, 1, dw_stride=trans_dw_stride // 4, embed_dim=embed_dim,
                                           num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                           qk_scale=qk_scale,
                                           drop_rate=drop_rate, attn_drop_rate=attn_drop_rate)
        # decoder
        self.Up1 = up_conv(512, 256)
        self.Up1_conv = conv_block(576, 256)
        self.Up2 = up_conv(256, 128)
        self.Up2_conv = conv_block(288, 128)
        self.Up3 = up_conv(128, 64)
        self.Up3_conv = conv_block(128, 64)
        self.seg = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 1)
        )
        # self.multi42 = F3F(cl=128,cx=64,ch=512)
        # self.multi84 = F3F(cl=64, cx=32, ch=256)
        # self.dspp = DSPP(in_channels=512,out_ch=512)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
    # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        # stem stage [N, 3, 336, 336] -> [N, 64, 84, 84]
        x168 = self.maxpool1(self.Conv1(x))  # [bs,64,168,168]
        x_base = self.maxpool2(self.Conv2(x168)) # [bs,64,84,84]
        # x_base = self.maxpool(self.act1(self.bn1(self.conv1(x))))
        x_t = self.trans_patch_conv(x_base).flatten(2)
        x_t = self.trans_1(x_t)  # [2, 441, 441]
        x = self.conv_1(x_base, return_x_2=False)  # [2, 128, 84, 84]
        x84,x_t_r84,x_t = self.first_stage(x,x_t)
        x84, x_t_r84, x_t = self.first_stage2(x84, x_t)
        # x, x_t_r, x_t = self.first_stage2(x, x_t)
        # x [2, 128, 84, 84] | x_t[2, 441, 441] | x_t_r[2, 32, 84, 84]
        # print("x shape = {} | x_t shape ={} | x_t_r shape = {}".format(x.shape,x_t.shape,x_t_r.shape))
        x42 = self.second_conv(self.second_pool(x84))
        # [2, 256, 42, 42]
        x42, x_t_r42, x_t = self.second_stage(x42, x_t)
        x42, x_t_r42, x_t = self.second_stage2(x42, x_t)
        x42, x_t_r42, x_t = self.second_stage3(x42, x_t)
        # x->[2, 256, 42, 42] | x_t->[2, 441, 441]| x_t_r->[2, 64, 42, 42]
        x21 = self.third_conv(self.third_pool(x42))
        # x->[2, 512, 21, 21]
        x21, x_t_r21, x_t = self.third_stage(x21, x_t)
        x21, x_t_r21, x_t = self.third_stage2(x21, x_t)
        x21, x_t_r21, x_t = self.third_stage3(x21, x_t)
        # x->[2, 512, 21, 21] | x_t->[2, 441, 441]| x_t_r->[2, 128, 21, 21]
        # print("x shape = {} | x_t shape ={} | x_t_r shape = {}".format(x.shape, x_t.shape, x_t_r.shape))
        #x = self.dspp(x)

        x = self.Up1(x21)  # [2,256,42,42]
        # x_t_r42 = self.multi42(x84,x_t_r42,x21)

        x = torch.cat([x, x42, x_t_r42], dim=1)  # [2,576,42,42]
        x = self.Up1_conv(x)  # [2,256,42,42]
        x = self.Up2(x)  # [2,128,84,84]
        # print("x84 = {}| x_t_r42 ={} |x21 ={}".format(x168.shape, x_t_r84.shape, x42.shape))
        # x_t_r84 = self.multi84(x168, x_t_r84, x42)
        x = torch.cat([x, x84, x_t_r84], dim=1)  # [2,288,84,84]
        x = self.Up2_conv(x)  # [2,128,84,84]
        x = self.Up3(x)       # [2,64,168,168]
        x = torch.cat([x,x168],dim=1) # [2,128,168,168]
        x = self.Up3_conv(x)
        x = F.interpolate(x, size=(336, 336), mode='bilinear', align_corners=False)
        x = self.seg(x)
        return x


"""input = torch.randn(2,3,336,336)
model = U_Net_DSPP_Transformer(get_type="mix")
out = model(input)
print(out.shape)
from utilities.utils import  count_params
print(count_params(model))"""