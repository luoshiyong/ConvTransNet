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
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        print("q shape = ",q.shape)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # print("********************attention shape = ",attn.shape)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
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
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
class FCUDown(nn.Module):
    """ CNN feature maps -> Transformer patch embeddings
    """

    def __init__(self, inplanes, outplanes, dw_stride,type,act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(FCUDown, self).__init__()
        self.dw_stride = dw_stride
        self.type = type
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.sample_pooling = nn.AvgPool2d(kernel_size=dw_stride, stride=dw_stride)
        self.local_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (c h w) (p1 p2)', p1=21, p2=21),
            # nn.Linear(self.dim*self.dim, self.dim*self.dim),
        )
        self.ln = norm_layer(441)
        self.ln_local = norm_layer(441)
        self.act = act_layer()

    def forward(self, x, x_t=None):
        x = self.conv_project(x)  # [N, C, H, W]
        # [N,C,H,W]->[N,C,441]
        if self.type=="global":
            x = self.sample_pooling(x).flatten(2)
            x = self.ln(x)
            x = self.act(x)
            return x
        elif self.type=="local":
            x = self.local_embedding(x)
            print("local embeding shape = ",x.shape)
            x = self.ln_local(x)
            x = self.act(x)
            return x

        else:
            return x


class FCUUp(nn.Module):
    """ Transformer patch embeddings -> CNN feature maps
    """

    def __init__(self, inplanes, outplanes, up_stride,type,act_layer=nn.ReLU,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6),):
        super(FCUUp, self).__init__()
        self.type = type
        self.up_stride = up_stride
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.bn = norm_layer(outplanes)
        self.act = act_layer()
        self.res_out = Rearrange('b (c p1 p2) (w1 w2) -> b c (p1  w1) (p2 w2)', p1=up_stride, p2=up_stride,
                                 w1=21, w2=21)
    def forward(self, x):
        bb,cc,_ = x.shape
        # [N, 384, 441] -> [N, 384, 21, 21]
        if self.type == "global":
            x_r = x.reshape(bb, cc, 21,21)
            x_r = self.act(self.bn(self.conv_project(x_r)))
            return F.interpolate(x_r, size=(21* self.up_stride, 21 * self.up_stride))
        elif self.type == "local":
            x_r = self.res_out(x)
            return x_r

class LTB(nn.Module):
    def __init__(self,in_ch,out_ch,size = 84,down_ratio=1):
        super(LTB, self).__init__()
        self.down_ratio = down_ratio
        self.dim = size//down_ratio
        self.fcudown = FCUDown(in_ch,in_ch,dw_stride=down_ratio,type="local")
        self.fcuup = FCUUp(in_ch,in_ch,up_stride=down_ratio,type="local")
        self.block = Block(dim=441,num_heads=9)
        """self.project = nn.Sequential(
            nn.Conv2d(in_ch*2, out_ch, 3,padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))"""
    def forward(self,x):
        res = x    # residual
        bb, cc, ww, hh = x.shape
        # local branch
        x = self.fcudown(x)
        x = self.block(x)
        x = self.fcuup(x)
        # x = torch.cat([res,x],dim=1)
        # x = self.project(x)
        return x+res

class GTB(nn.Module):
    def __init__(self,in_ch,out_ch,size = 84,down_ratio=1):
        super(GTB, self).__init__()
        self.down_ratio = down_ratio
        self.dim = size//down_ratio
        self.size = size
        """self.global_embedding = nn.Sequential(
            nn.Linear(self.size*self.size, 441)
        )"""
        self.fcudown = FCUDown(in_ch, in_ch, dw_stride=down_ratio, type="global")
        self.fcuup = FCUUp(in_ch, in_ch, up_stride=down_ratio, type="global")
        self.block = Block(dim=441, num_heads=9)
        # self.res_out = Rearrange('b (c p1 p2) (w1 w2) -> b c (p1  w1) (p2 w2)',p1=down_ratio,p2=down_ratio,w1=self.dim,w2=self.dim)
        self.attention_norm = LayerNorm(self.dim*self.dim, eps=1e-6)
        # self.ffn_norm = LayerNorm(441, eps=1e-6)
        self.upsampe = nn.Upsample(scale_factor=down_ratio,mode='bilinear')
        self.upnorm = nn.BatchNorm2d(out_ch)
        # self.ffn = Mlp(in_features=441,out_features=441)
        """self.project = nn.Sequential(
            nn.Conv2d(in_ch*2, out_ch, 3,padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))"""
    def forward(self,x):
        res = x    # residual
        # local branch
        x = self.fcudown(x)  # [2,64,441]
        x = x + self.block(x)  #[2,64,441]
        x = self.fcuup(x)
        print("x shape = {} | res shape ={}".format(x.shape,res.shape))
        return x+res
class HTB(nn.Module):
    def __init__(self,in_ch,out_ch,size = 84,down_ratio=1):
        super(HTB, self).__init__()
        self.down_ratio = down_ratio
        self.dim = size//down_ratio
        self.size = size
        self.global_reduce = nn.Sequential(
            nn.Linear(441, down_ratio*down_ratio)
        )
        self.global_embedding = nn.Conv2d(in_ch, in_ch, 3, padding=1, stride=down_ratio)
        self.local_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (c h w) (p1 p2)', p1=self.dim, p2=self.dim),
            # nn.Linear(self.dim * self.dim, self.dim * self.dim),
        )
        self.res_out = Rearrange('b (c p1 p2) (w1 w2) -> b c (p1  w1) (p2 w2)',p1=down_ratio,p2=down_ratio,w1=self.dim,w2=self.dim)
        self.attention_norm = LayerNorm(self.dim*self.dim, eps=1e-6)
        # self.ffn_norm = LayerNorm(441, eps=1e-6)
        self.ffn_normxx = LayerNorm(441+self.down_ratio*self.down_ratio, eps=1e-6)
        self.attention = nn.MultiheadAttention(embed_dim=441, num_heads=9, dropout=0.1)
        self.ffnxx = Mlp(in_features=441+self.down_ratio*self.down_ratio, out_features=441)
        self.resnorm = nn.BatchNorm2d(out_ch)
        # self.ffn = Mlp(in_features=441,out_features=441)
        """self.project = nn.Sequential(
            nn.Conv2d(in_ch*2, out_ch, 3,padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))"""
    def forward(self,x):
        res = x    # residual
        bb, cc, ww, hh = x.shape
        # local branch
        x = self.global_embedding(x)  # [bs,c,n]
        x = x.flatten(2)
        x = self.global_reduce(x)  # [bs,c,4]
        x = x.repeat(1,self.down_ratio*self.down_ratio,1)
        xlocal = self.local_embedding(res) # [bs,gg*n]
        x = torch.cat([x,xlocal],dim=-1)
        # print("xxxx shape = ",x.shape)
        x = self.ffn_normxx(x)
        x = self.ffnxx(x)
        x = self.attention_norm(x)
        x = x + self.attention(x.transpose(1,0),x.transpose(1,0),x.transpose(1,0))[0].transpose(1,0)
        # x = self.ffn_normxx(x)
        # x = self.ffnxx(x)
        x = self.resnorm(self.res_out(x))
        # x = torch.cat([res,x],dim=1)
        # x = self.project(x)
        return x+res
class TLTB(nn.Module):
    def __init__(self,in_ch,out_ch,size = 84,down_ratio=1,ttype=1):
        super(TLTB, self).__init__()
        self.ttype = ttype
        self.local_atten = LTB(in_ch=in_ch,out_ch=out_ch,size=size,down_ratio=down_ratio)
        self.global_atten = GTB(in_ch=in_ch,out_ch=out_ch,size=size,down_ratio=down_ratio)
        # self.hybrid_atten = HTB(in_ch=in_ch,out_ch=out_ch,size=size,down_ratio=down_ratio)
        self.norm1 = nn.BatchNorm2d(out_ch)
        self.norm2 = nn.BatchNorm2d(out_ch)
        self.norm3 = nn.BatchNorm2d(out_ch)

        self.outconv =  self.project = nn.Sequential(
            nn.Conv2d(out_ch*4, out_ch, 3,padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self,x):
        xlocal = self.local_atten(x)
        # print("local x shape = ",xlocal.shape)
        xgloabl = self.global_atten(x)
        # print("global x shape = ", xglobal.shape)
        xhybrid = self.global_atten(xlocal)
        # print("local x shape = ", xhybrid.shape)
        return self.project(torch.cat([x,xlocal,xgloabl,xhybrid],dim=1))
class U_Net_DSPP_Transformer(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, args):
        super(U_Net_DSPP_Transformer, self).__init__()
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

        # self.dspp1 = DSPP(in_channels=256, out_ch=256)
        # self.dspp2 = DSPP(in_channels=128, out_ch=128)
        # self.dspp3 = DSPP(in_channels=64, out_ch=64)
        self.mstb3 = TLTB(in_ch=128,out_ch=128,size=84,down_ratio=4)
        self.mstb2 = TLTB(in_ch=256,out_ch=256,size=42,down_ratio=2)
        self.mstb1 = LTB(in_ch=512, out_ch=512, size=21, down_ratio=1)
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

        # e4 = self.dspp1(e4)  # [b, 256, 42,42]
        # e3 = self.dspp2(e3)  # [b,128, 84,84]
        # e2 = self.dspp3(e2)  # [b,64, 168, 168]

        e5 = self.mstb1(e5)
        e4 = self.mstb2(e4)
        e3 = self.mstb3(e3)

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

input = torch.randn(2,64,84,84)
model = TLTB(in_ch=64,out_ch=64,size=84,down_ratio=4)
# input = torch.randn(2,128,42,42)
# model = GTB(in_ch=128,out_ch=128,size=42,down_ratio=2)
# model = TLTB(in_ch=128,out_ch=128,size=42,down_ratio=2)
# local = model.local_atten
# glo = model.global_atten
# hybrid = model.hybrid_atten

# model = U_Net_DSPP('')
# model = U_Net('')
out = model(input)

from utilities.utils import  count_params
print(count_params(model))
# print("local = {} | global = {}|hybrid ={}".format(count_params(local),count_params(glo),count_params(hybrid)))
# torch.save(model.state_dict(),"U:/paper3/paper_pipeline/unet64.pth")"""



"""input = torch.randn(2,3,336,336)
model = U_Net_DSPP_Transformer("")
out = model(input)
from utilities.utils import  count_params
print(count_params(model))"""