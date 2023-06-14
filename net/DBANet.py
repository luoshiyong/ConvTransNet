from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from torch import Tensor
from typing import Tuple, Optional
def Conv1x1BN(in_channels,out_channels, stride=1, groups=1, bias=False):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, groups=groups, bias=bias),
            nn.BatchNorm2d(out_channels)
        )

def Conv3x3BN(in_channels,out_channels, stride=1, groups=1, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=stride,padding=1, groups=groups, bias=bias),
        nn.BatchNorm2d(out_channels)
    )

class DEPTHWISECONV(nn.Module):
    def __init__(self,in_ch,out_ch,kernel = 3,padding = 1):
        super(DEPTHWISECONV, self).__init__()
        # 也相当于分组为1的分组卷积
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=kernel,
                                    stride=1,
                                    padding=padding,
                                    groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
    def forward(self,input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=8):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)				# 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch,group = 1):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True, groups=group),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True, groups=group),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
            )

    def forward(self, x):

        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch,group = 1):
        super(up_conv, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True,groups=group),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.up(x)
        return x


class ConvGRU(nn.Module):
    def __init__(self,
                 channels: int,
                 kernel_size: int = 3,
                 padding: int = 1):
        super().__init__()
        channels = channels//3
        self.channels = channels
        self.ih = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, kernel_size, padding=padding),
            nn.Sigmoid()
        )
        self.hh = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size, padding=padding),
            nn.Tanh()
        )

    def forward_single_frame(self, x, h):
        r, z = self.ih(torch.cat([x, h], dim=1)).split(self.channels, dim=1)
        c = self.hh(torch.cat([x, r * h], dim=1))
        h = (1 - z) * h + z * c
        return h, h

    def forward_time_series(self, x, h):
        o = []
        for xt in x.unbind(dim=1):
            ot, h = self.forward_single_frame(xt, h)
            o.append(ot)
        o = torch.stack(o, dim=1)
        return o, h

    def forward(self, x, h: Optional[Tensor]):
        if h is None:
            # print("x size = ",x.size())
            h = torch.zeros((x.size(0), x.size(-3)//3, x.size(-2), x.size(-1)),
                            device=x.device, dtype=x.dtype)
        # x->  [4, 120, 168, 168]->[4,3,40,168,168]
        bb,cc,hh,ww = x.shape
        x = x.reshape(bb,3,cc//3,hh,ww)
        if x.ndim == 5:
            x,ps = self.forward_time_series(x, h)
            # print("ndim 5 out shape = ",x.shape)
            return x.reshape(bb,cc,hh,ww),ps
        else:
            return self.forward_single_frame(x, h)

class BottleneckBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.gru = ConvGRU(channels)

    def forward(self, x, r: Optional[Tensor] = None):
        b, r = self.gru(x, r)
        return b, r

def CConcat(f1,f2,group):
    assert f1.shape[2]==f1.shape[2] and f1.shape[3]==f2.shape[3],"CConcat shape not match!"
    outf1 = f1.split(int(f1.shape[1]/group),dim=1)
    outf2 = f2.split(int(f2.shape[1]/group),dim=1)
    out = []
    for idx in range(len(outf1)):
        cur = torch.cat((outf1[idx],outf2[idx]),dim=1)
        out.append(cur)
    return torch.cat(out,dim=1)
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation,group=1):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False,groups=group),
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
def CConcat(f1,f2,group):
    assert f1.shape[2]==f1.shape[2] and f1.shape[3]==f2.shape[3],"CConcat shape not match!"
    outf1 = f1.split(int(f1.shape[1]/group),dim=1)
    outf2 = f2.split(int(f2.shape[1]/group),dim=1)
    out = []
    for idx in range(len(outf1)):
        cur = torch.cat((outf1[idx],outf2[idx]),dim=1)
        out.append(cur)
    return torch.cat(out,dim=1)
class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates,group = 1):
        super(ASPP, self).__init__()
        out_channels = 960
        modules = []
        self.group = group
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False,groups=group),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1,group))
        modules.append(ASPPConv(in_channels, out_channels, rate2,group))
        modules.append(ASPPConv(in_channels, out_channels, rate3,group))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False,groups=group),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        idx = 0
        for conv in self.convs:
            if idx == 0:
                out = conv(x)
            else:
                out = CConcat(out,conv(x),self.group)
            idx += 1
        return self.project(out)
"""
    Deformable Transformer Block 
"""
from net.defconv import DefC
from timm.models.layers import DropPath, trunc_normal_
from functools import partial
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return DefC(in_planes, out_planes, kernel_size=3, stride=stride,padding=dilation, bias=False)
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

"""
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.toq = conv3x3(960,480,1)
        self.tok = conv3x3(960, 480, 1)
        self.tov = conv3x3(960, 480, 1)
        self.res = nn.Conv2d(480,960,3,padding=1)
        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        BB, NN, CC = x.shape
        x = x.reshape(BB,NN,21,21)
        B,C,H,W = x.shape
        q = self.toq(x).flatten(2).reshape(B, C//2, self.num_heads, H*W // self.num_heads).permute(0, 2, 1, 3)
        k = self.tok(x).flatten(2).reshape(B, C//2, self.num_heads, H*W // self.num_heads).permute(0, 2, 1, 3)
        v = self.tov(x).flatten(2).reshape(B, C//2, self.num_heads, H*W // self.num_heads).permute(0, 2, 1, 3)
        # print("k shape = ",k.shape)
        # B, N, C = x.shape
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, C//2, H*W)
        # print("x shape = ",x.shape)   # [4, 480, 441]
        x = self.proj(x)
        x = self.proj_drop(x)
        x = self.res(x.reshape(B,C//2,H,W)).flatten(2)
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
        B,C,H,W = x.shape
        x = x.reshape(B,C,H*W)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x.reshape(B,C,H,W)"""
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d,LayerNorm
import math
class Block(nn.Module):
    def __init__(self, hidden_size=441):
        super(Block, self).__init__()
        self.hidden_size = hidden_size
        self.attention_norm = LayerNorm(hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(hidden_size, eps=1e-6)
        self.ffn = Mlp(in_features=441)
        self.attn = Attention()

    def forward(self, x):
        B,C,_= x.shape
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return  x.reshape(B,C,21,21)

class Attention(nn.Module):
    def __init__(self, hidden_size=441,num_heads=9,attention_dropout_rate=0.1,vis=False):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = num_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(hidden_size, self.all_head_size)
        self.key = Linear(hidden_size, self.all_head_size)
        self.value = Linear(hidden_size, self.all_head_size)

        self.out = Linear(hidden_size, hidden_size)
        self.attn_dropout = Dropout(attention_dropout_rate)
        self.proj_dropout = Dropout(attention_dropout_rate)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output


class Single_Branceg(nn.Module):    # all use group conv
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, group = 1):
        super(Single_Branceg, self).__init__()
        in_ch = 5
        out_ch = group
        n1 = 60
        self.group = group
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.aspp_dilate = [12, 24, 36]
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(in_ch, filters[0])
        self.gru1 = BottleneckBlock(filters[0])
        self.Conv2 = conv_block(filters[0], filters[1],group)
        self.gru2 = BottleneckBlock(filters[1])
        self.Conv3 = conv_block(filters[1], filters[2],group)
        self.gru3 = BottleneckBlock(filters[2])
        self.Conv4 = conv_block(filters[2], filters[3],group)
        self.gru4 = BottleneckBlock(filters[3])
        self.Conv5 = conv_block(filters[3], filters[4],group)
        # self.gru5 = BottleneckBlock(filters[4])
        self.gaspp = ASPP(filters[4],self.aspp_dilate,group)
        self.Up5 = up_conv(filters[4], filters[3],group)
        self.Up_conv5 = conv_block(filters[4], filters[3],group)

        self.Up4 = up_conv(filters[3], filters[2],group)
        self.Up_conv4 = conv_block(filters[3], filters[2],group)

        self.Up3 = up_conv(filters[2], filters[1],group)
        self.Up_conv3 = conv_block(filters[2], filters[1],group)

        self.Up2 = up_conv(filters[1], filters[0],group)
        self.Up_conv2 = conv_block(filters[1], filters[0],group)

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

       # self.active = torch.nn.Sigmoid()

    def forward(self, x):

        e1 = self.Conv1(x)
        e1,r = self.gru1(e1)
        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)
        e2,r = self.gru2(e2)
        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)
        e3,r = self.gru3(e3)
        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)
        e4,r = self.gru4(e4)
        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)
        # print("before aspp shape = ",e5.shape) # [4, 960, 21, 21]
        e5 = self.gaspp(e5)

        d5 = self.Up5(e5)
        # d5 = torch.cat((e4, d5), dim=1)
        d5 = CConcat(e4, d5, self.group)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        # d4 = torch.cat((e3, d4), dim=1)
        d4 = CConcat(e3, d4, self.group)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        # d3 = torch.cat((e2, d3), dim=1)
        d3 = CConcat(e2, d3, self.group)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        # d2 = torch.cat((e1, d2), dim=1)
        d2 = CConcat(e1, d2, self.group)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        #d1 = self.active(out)

        return out


class Baseline(nn.Module):    # all use group conv
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, group = 1):
        super(Baseline, self).__init__()
        self.group = group
        in_ch = 3
        out_ch = 3
        n1 = 60
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.aspp_dilate = [12, 24, 36]
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(in_ch, filters[0])
        self.gru1 = BottleneckBlock(filters[0])
        self.Conv2 = conv_block(filters[0], filters[1],group)
        self.gru2 = BottleneckBlock(filters[1])
        self.Conv3 = conv_block(filters[1], filters[2],group)
        self.gru3 = BottleneckBlock(filters[2])
        self.Conv4 = conv_block(filters[2], filters[3],group)
        self.gru4 = BottleneckBlock(filters[3])
        self.Conv5 = conv_block(filters[3], filters[4],group)
        # self.gru5 = BottleneckBlock(filters[4])
        self.gaspp = ASPP(filters[4],self.aspp_dilate,group)
        # branceg
        self.Up5g = up_conv(filters[4], filters[3],group)
        self.Up_conv5g = conv_block(filters[4], filters[3],group)

        self.Up4g = up_conv(filters[3], filters[2],group)
        self.Up_conv4g = conv_block(filters[3], filters[2],group)

        self.Up3g = up_conv(filters[2], filters[1],group)
        self.Up_conv3g = conv_block(filters[2], filters[1],group)

        self.Up2g = up_conv(filters[1], filters[0],group)
        self.Up_conv2g = conv_block(filters[1], filters[0],group)
        # brancec
        self.Up5c = up_conv(filters[4], filters[3])
        self.Up_conv5c = conv_block(filters[4], filters[3])
        self.Up4c = up_conv(filters[3], filters[2])
        self.Up_conv4c = conv_block(filters[3], filters[2])
        self.Up3c = up_conv(filters[2], filters[1])
        self.Up_conv3c = conv_block(filters[2], filters[1])
        self.Up2c = up_conv(filters[1], filters[0])
        self.Up_conv2c = conv_block(filters[1], filters[0])
        self.Conv = nn.Sequential(
            nn.Conv2d(filters[0]*2, filters[0]//2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(filters[0]//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[0]//2, out_ch, kernel_size=1)
        )
        # self.Conv = nn.Conv2d(filters[0]*2, out_ch, kernel_size=1, stride=1, padding=0)

       # self.active = torch.nn.Sigmoid()

    def forward(self, x):

        e1 = self.Conv1(x)
        e1,r = self.gru1(e1)
        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)
        e2,r = self.gru2(e2)
        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)
        e3,r = self.gru3(e3)
        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)
        e4,r = self.gru4(e4)
        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)
        # print("before aspp shape = ",e5.shape) # [4, 960, 21, 21]
        e5 = self.gaspp(e5)

        d5c = self.Up5c(e5)
        d5c = torch.cat((e4, d5c), dim=1)

        d5c = self.Up_conv5c(d5c)

        d4c = self.Up4c(d5c)
        d4c = torch.cat((e3, d4c), dim=1)
        d4c = self.Up_conv4c(d4c)

        d3c = self.Up3c(d4c)
        d3c = torch.cat((e2, d3c), dim=1)
        d3c = self.Up_conv3c(d3c)

        d2c = self.Up2c(d3c)
        d2c = torch.cat((e1, d2c), dim=1)
        d2c = self.Up_conv2c(d2c)
        # print("brance c out shape = ",d2c.shape)
        d5g = self.Up5g(e5)
        # d5 = torch.cat((e4, d5), dim=1)
        d5g = CConcat(e4, d5g, self.group)
        d5g = self.Up_conv5g(d5g)

        d4g = self.Up4g(d5g)
        # d4 = torch.cat((e3, d4), dim=1)
        d4g = CConcat(e3, d4g, self.group)
        d4g = self.Up_conv4g(d4g)

        d3g = self.Up3g(d4g)
        # d3 = torch.cat((e2, d3), dim=1)
        d3g = CConcat(e2, d3g, self.group)
        d3g = self.Up_conv3g(d3g)

        d2g = self.Up2g(d3g)
        # d2 = torch.cat((e1, d2), dim=1)
        d2g = CConcat(e1, d2g, self.group)
        d2g = self.Up_conv2g(d2g)
        # print("brance g out shape = ",d2g.shape)

        out = self.Conv(torch.cat([d2g,d2c],dim=1))

        #d1 = self.active(out)

        return out
class CBFM(nn.Module):    # all use group conv
    def __init__(self, group = 1,in_ch=512):
        super(CBFM, self).__init__()
        self.group = group
        self.cf1  = nn.Sequential(
            nn.Conv2d(in_ch, int(in_ch/10), kernel_size=3, stride=1, padding=1, bias=True, groups=group),
            nn.BatchNorm2d(int(in_ch/10)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_ch / 10),group,kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        reconv = []
        for i in range(self.group):
            reconv.append(nn.Conv2d(in_ch,int(in_ch / self.group),kernel_size=1, bias=True))
        self.reconvs = nn.ModuleList(reconv)
        self.gfout = conv_block(in_ch*2,in_ch,group = self.group)
        self.se = SE_Block(in_ch)
        self.xx3 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=True, groups=group),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        )
        self.cfout = conv_block(in_ch*2,in_ch)
    def forward(self,cf,gf):
        # print("cf shaep = ", cf.shape)
        # print("gf shape = ",gf.shape)
        ac1 = self.cf1(cf)
        cfout = []
        for i in range(self.group):
            # print("ac shape = ",ac1[:,[i],:,:].shape)
            # print("cf shaep = ",cf.shape)
            cfout.append(self.reconvs[i](ac1[:,[i],:,:]*cf))
        out_gf = self.gfout(CConcat(gf,torch.cat(cfout,dim=1),self.group))
        out_cf = self.cfout(torch.cat([self.xx3(self.se(gf)),cf],dim=1))
        return out_cf,out_gf

class DBANet(nn.Module):    # all use group conv
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, group = 1):
        super(DBANet, self).__init__()
        self.group = group
        in_ch = 3
        out_ch = 2
        n1 = 60
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.aspp_dilate = [12, 24, 36]
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(in_ch, filters[0])
        self.gru1 = BottleneckBlock(filters[0])
        self.Conv2 = conv_block(filters[0], filters[1],group)
        self.gru2 = BottleneckBlock(filters[1])
        self.Conv3 = conv_block(filters[1], filters[2],group)
        self.gru3 = BottleneckBlock(filters[2])
        self.Conv4 = conv_block(filters[2], filters[3],group)
        self.gru4 = BottleneckBlock(filters[3])
        self.Conv5 = conv_block(filters[3], filters[4],group)
        # self.gru5 = BottleneckBlock(filters[4])
        self.mst = Block()
        # branceg
        self.Up5g = up_conv(filters[4], filters[3],group)
        self.Up_conv5g = conv_block(filters[4], filters[3],group)

        self.Up4g = up_conv(filters[3], filters[2],group)
        self.Up_conv4g = conv_block(filters[3], filters[2],group)

        self.Up3g = up_conv(filters[2], filters[1],group)
        self.Up_conv3g = conv_block(filters[2], filters[1],group)

        self.Up2g = up_conv(filters[1], filters[0],group)
        self.Up_conv2g = conv_block(filters[1], filters[0],group)
        # brancec
        self.Up5c = up_conv(filters[4], filters[3])
        self.Up_conv5c = conv_block(filters[4], filters[3])
        self.Up4c = up_conv(filters[3], filters[2])
        self.Up_conv4c = conv_block(filters[3], filters[2])
        self.Up3c = up_conv(filters[2], filters[1])
        self.Up_conv3c = conv_block(filters[2], filters[1])
        self.Up2c = up_conv(filters[1], filters[0])
        self.Up_conv2c = conv_block(filters[1], filters[0])

        # self.CBFM1 = CBFM(group,filters[3])
        self.CBFM2 = CBFM(group, filters[2])
        self.CBFM3 = CBFM(group, filters[1])
        # self.CBFM4 = CBFM(group, filters[0])

        self.Conv = nn.Conv2d(filters[0]*2, out_ch, kernel_size=1, stride=1, padding=0)

       # self.active = torch.nn.Sigmoid()

    def forward(self, x):

        e1 = self.Conv1(x)
        e1,r = self.gru1(e1)
        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)
        e2,r = self.gru2(e2)
        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)
        e3,r = self.gru3(e3)
        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)
        e4,r = self.gru4(e4)
        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)
        # print("before aspp shape = ",e5.shape) # [4, 960, 21, 21]
        e5 = e5.flatten(2)
        e5 = self.mst(e5)
        # e5,r = self.gru5(e5)
        # Decoder ==============================================================================
        d5c = self.Up5c(e5)
        d5c = torch.cat((e4, d5c), dim=1)
        d5c = self.Up_conv5c(d5c)
        d5g = self.Up5g(e5)
        # d5 = torch.cat((e4, d5), dim=1)
        d5g = CConcat(e4, d5g, self.group)
        d5g = self.Up_conv5g(d5g)
        # d5c,d5g = self.CBFM1(d5c,d5g)


        d4c = self.Up4c(d5c)
        d4c = torch.cat((e3, d4c), dim=1)
        d4c = self.Up_conv4c(d4c)
        d4g = self.Up4g(d5g)
        # d4 = torch.cat((e3, d4), dim=1)
        d4g = CConcat(e3, d4g, self.group)
        d4g = self.Up_conv4g(d4g)
        d4c,d4g = self.CBFM2(d4c,d4g)

        d3c = self.Up3c(d4c)
        d3c = torch.cat((e2, d3c), dim=1)
        d3c = self.Up_conv3c(d3c)
        d3g = self.Up3g(d4g)
        # d3 = torch.cat((e2, d3), dim=1)
        d3g = CConcat(e2, d3g, self.group)
        d3g = self.Up_conv3g(d3g)
        d3c,d3g = self.CBFM3(d3c,d3g)

        d2c = self.Up2c(d3c)
        d2c = torch.cat((e1, d2c), dim=1)
        d2c = self.Up_conv2c(d2c)
        # print("brance c out shape = ",d2c.shape)
        d2g = self.Up2g(d3g)
        # d2 = torch.cat((e1, d2), dim=1)
        d2g = CConcat(e1, d2g, self.group)
        d2g = self.Up_conv2g(d2g)
        # print("brance g out shape = ",d2g.shape)
        # d2c,d2g = self.CBFM4(d2c,d2g)

        out = self.Conv(torch.cat([d2g,d2c],dim=1))

        #d1 = self.active(out)

        return out
class DBANet53(nn.Module):    # all use group conv
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, group = 1):
        super(DBANet53, self).__init__()
        self.group = group
        in_ch = 5
        out_ch = 6
        n1 = 60
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        # self.aspp_dilate = [12, 24, 36]
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(in_ch, filters[0])
        self.gru1 = BottleneckBlock(filters[0])
        self.Conv2 = conv_block(filters[0], filters[1],group)
        self.gru2 = BottleneckBlock(filters[1])
        self.Conv3 = conv_block(filters[1], filters[2],group)
        self.gru3 = BottleneckBlock(filters[2])
        self.Conv4 = conv_block(filters[2], filters[3],group)
        self.gru4 = BottleneckBlock(filters[3])
        self.Conv5 = conv_block(filters[3], filters[4],group)
        # self.gru5 = BottleneckBlock(filters[4])
        # self.mst = Block()
        # branceg
        self.Up5g = up_conv(filters[4], filters[3],group)
        self.Up_conv5g = conv_block(filters[4], filters[3],group)

        self.Up4g = up_conv(filters[3], filters[2],group)
        self.Up_conv4g = conv_block(filters[3], filters[2],group)

        self.Up3g = up_conv(filters[2], filters[1],group)
        self.Up_conv3g = conv_block(filters[2], filters[1],group)

        self.Up2g = up_conv(filters[1], filters[0],group)
        self.Up_conv2g = conv_block(filters[1], filters[0],group)
        # brancec
        self.Up5c = up_conv(filters[4], filters[3])
        self.Up_conv5c = conv_block(filters[4], filters[3])
        self.Up4c = up_conv(filters[3], filters[2])
        self.Up_conv4c = conv_block(filters[3], filters[2])
        self.Up3c = up_conv(filters[2], filters[1])
        self.Up_conv3c = conv_block(filters[2], filters[1])
        self.Up2c = up_conv(filters[1], filters[0])
        self.Up_conv2c = conv_block(filters[1], filters[0])

        # self.CBFM1 = CBFM(group,filters[3])
        self.CBFM2 = CBFM(group, filters[2])
        self.CBFM3 = CBFM(group, filters[1])
        # self.CBFM4 = CBFM(group, filters[0])

        self.Conv = nn.Conv2d(filters[0]*2, out_ch, kernel_size=1, stride=1, padding=0)

       # self.active = torch.nn.Sigmoid()

    def forward(self, x):

        e1 = self.Conv1(x)
        e1,r = self.gru1(e1)
        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)
        e2,r = self.gru2(e2)
        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)
        e3,r = self.gru3(e3)
        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)
        e4,r = self.gru4(e4)
        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)
        # print("before aspp shape = ",e5.shape) # [4, 960, 21, 21]
        # e5 = e5.flatten(2)
        # e5 = self.mst(e5)
        # e5,r = self.gru5(e5)
        # Decoder ==============================================================================
        d5c = self.Up5c(e5)
        d5c = torch.cat((e4, d5c), dim=1)
        d5c = self.Up_conv5c(d5c)
        d5g = self.Up5g(e5)
        # d5 = torch.cat((e4, d5), dim=1)
        d5g = CConcat(e4, d5g, self.group)
        d5g = self.Up_conv5g(d5g)
        # d5c,d5g = self.CBFM1(d5c,d5g)


        d4c = self.Up4c(d5c)
        d4c = torch.cat((e3, d4c), dim=1)
        d4c = self.Up_conv4c(d4c)
        d4g = self.Up4g(d5g)
        # d4 = torch.cat((e3, d4), dim=1)
        d4g = CConcat(e3, d4g, self.group)
        d4g = self.Up_conv4g(d4g)
        d4c,d4g = self.CBFM2(d4c,d4g)

        d3c = self.Up3c(d4c)
        d3c = torch.cat((e2, d3c), dim=1)
        d3c = self.Up_conv3c(d3c)
        d3g = self.Up3g(d4g)
        # d3 = torch.cat((e2, d3), dim=1)
        d3g = CConcat(e2, d3g, self.group)
        d3g = self.Up_conv3g(d3g)
        d3c,d3g = self.CBFM3(d3c,d3g)

        d2c = self.Up2c(d3c)
        d2c = torch.cat((e1, d2c), dim=1)
        d2c = self.Up_conv2c(d2c)
        # print("brance c out shape = ",d2c.shape)
        d2g = self.Up2g(d3g)
        # d2 = torch.cat((e1, d2), dim=1)
        d2g = CConcat(e1, d2g, self.group)
        d2g = self.Up_conv2g(d2g)
        # print("brance g out shape = ",d2g.shape)
        # d2c,d2g = self.CBFM4(d2c,d2g)

        out = self.Conv(torch.cat([d2g,d2c],dim=1))

        #d1 = self.active(out)

        return out
class Single_Brance(nn.Module):    # all use group conv
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, group = 1):
        super(Single_Brance, self).__init__()
        in_ch = 5
        out_ch = group
        n1 = 60
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.aspp_dilate = [12, 24, 36]
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(in_ch, filters[0])
        self.gru1 = BottleneckBlock(filters[0])
        self.Conv2 = conv_block(filters[0], filters[1],group)
        self.gru2 = BottleneckBlock(filters[1])
        self.Conv3 = conv_block(filters[1], filters[2],group)
        self.gru3 = BottleneckBlock(filters[2])
        self.Conv4 = conv_block(filters[2], filters[3],group)
        self.gru4 = BottleneckBlock(filters[3])
        self.Conv5 = conv_block(filters[3], filters[4],group)
        # self.gru5 = BottleneckBlock(filters[4])
        self.gaspp = ASPP(filters[4],self.aspp_dilate,group)
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
        e1,r = self.gru1(e1)
        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)
        e2,r = self.gru2(e2)
        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)
        e3,r = self.gru3(e3)
        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)
        e4,r = self.gru4(e4)
        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)
        # print("before aspp shape = ",e5.shape) # [4, 960, 21, 21]
        e5 = self.gaspp(e5)

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

        #d1 = self.active(out)

        return out
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

        n1 =60
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
"""input = torch.randn(4,3,336,336)
model = DBANet(3)
# model = Single_Branceg(3)
out = model(input)
print("out shape = ",out.shape)
# model1 = Single_Branceg(3)
# model2 = Single_Brance(3)
print("model1 net parames = ",sum(param.numel() for param in model.parameters()))
# print("model1 net parames = ",sum(param.numel() for param in model1.parameters()))
# print("model2 net parames = ",sum(param.numel() for param in model2.parameters()))"""

#测试transunet-transformer
# model = Block()
#input = torch.randn(4,441,960)
# input = torch.randn(4,960,21,21)
# model = MSTransformer()
input = torch.randn(4,3,336,336)
model = DBANet(3)
# model = U_Net("")
out = model(input)
print(out.shape)
print("model1 net parames = ",sum(param.numel() for param in model.parameters()))
"""
SAU-Net:25766522
U-Net:
"""