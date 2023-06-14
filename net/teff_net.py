from net.defconv import DefC
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
import math
from functools import partial
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
# from timm.models.layers import DropPath,trunc_normal_
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




class DEPTHWISECONV(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DEPTHWISECONV, self).__init__()
        # 也相当于分组为1的分组卷积
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
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


class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                      dilation=dilation, bias=bias, groups=in_channels),
            # PointWise Conv
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )

        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


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
# [1,1,2,4,8]
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
            print("x.shape = {}|conv = {}".format(x[idx].shape,self.convs[idx]))
            res.append(self.convs[idx](x[idx]))
        res = torch.cat(res, dim=1)
        return self.project(torch.cat([res,y],dim=1))
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
    def __init__(self, num_heads=9,hidden_size=441,attention_dropout_rate=0.2,vis=False):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = num_heads                # 9
        self.attention_head_size = int(hidden_size / self.num_attention_heads) # 49
        self.all_head_size = self.num_attention_heads * self.attention_head_size # 441

        self.query = Linear(hidden_size, self.all_head_size)
        self.key = Linear(hidden_size, self.all_head_size)
        self.value = Linear(hidden_size, self.all_head_size)

        self.out = Linear(hidden_size, hidden_size)
        self.attn_dropout = Dropout(attention_dropout_rate)
        self.proj_dropout = Dropout(attention_dropout_rate)
        # self.scale_dim = [1,2,4,8,8,8,8,12,12]
        self.scale_dim = [0,1, 3, 7, 15, 23, 31, 39, 51, 63]
        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):   # [5,512,441]
        attention_list = []
        for i in range(len(self.scale_dim)-1):
            attention_list.append(x[:,:,self.scale_dim[i]*7:self.scale_dim[i+1]*7].unsqueeze(1))
            # print("dim = ",x[:,:,self.scale_dim[i]*7:self.scale_dim[i+1]*7].unsqueeze(1).shape)

        return attention_list

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states) # [bs,c,441]
        # multi-head attention
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        out_value = []
        weights = []
        # attention
        for (q,k,v) in zip(query_layer,key_layer,value_layer):
            attention_scores = torch.matmul(q, k.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(q.shape[-1])
            attention_probs = self.softmax(attention_scores)
            weights.append(attention_probs)  if self.vis else None  # [b,dim,dim]
            attention_probs = self.attn_dropout(attention_probs)

            context_layer = torch.matmul(attention_probs, v)  # [b,1,512,dim]
            context_layer = context_layer.permute(0, 2, 1, 3).squeeze().contiguous() # [b,512,dim]
            out_value.append(context_layer)
        out_value = torch.cat(out_value,dim=-1)
        attention_output = self.out(out_value)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


"""
Multi-Scale Transformer block
params:
------size:
"""
class MSTB(nn.Module):
    def __init__(self,in_ch,out_ch,down_ratio=1,out_format='conv',need_up=False):
        super(MSTB, self).__init__()
        self.need_up = need_up
        self.down_ratio = down_ratio
        self.out_format = out_format
        self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3,padding=1,stride=down_ratio,bias=False),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
        )
        self.attention_norm = LayerNorm(441, eps=1e-6)
        self.ffn_norm = LayerNorm(441, eps=1e-6)
        self.upsampe = nn.Upsample(scale_factor=down_ratio,mode='bilinear')
        self.attention = Attention()
        self.ffn = Mlp(in_features=441,out_features=441)
        self.project = nn.Sequential(
            nn.Conv2d(out_ch*2, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
    def forward(self,x):
        hh = x
        x = self.downsample(x)  # [bs,c,21,21]
        bb, cc, ww, hh = x.shape
        x = x.flatten(2)
        h = x
        x = self.attention_norm(x)
        x,_ = self.attention(x)
        x = x + h
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = h + x
        if self.out_format=='conv':
            x = x.view((bb,cc,ww,hh))
            if self.need_up:
                x = self.upsampe(x)
                x = torch.cat([hh,x],dim=1)
                x = self.project(x)
        return x
"""
Feature selective fusion module:
input: [b,c,441]
output:[b,c,441]
"""
class FSF(nn.Module):
    def __init__(self,low_ch,high_ch,up_rate =2):
        super(FSF, self).__init__()
        self.up_conv = nn.Upsample(scale_factor=up_rate)
        self.conv1 = nn.Sequential(
                nn.Conv2d(low_ch, low_ch, 3,padding=1,bias=False),
                nn.BatchNorm2d(low_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(low_ch, high_ch, 1,bias=False)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(high_ch, high_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(high_ch),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(high_ch, high_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(high_ch),
            nn.ReLU(inplace=True)
        )
        self.multihead = nn.MultiheadAttention(441,9,0.2)
        self.outconv = nn.Sequential(
            nn.Conv2d(high_ch*2, high_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(high_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self,lowf,highf):
        # lowf->k,v->main   [bs,512,21,21]
        # highf->q          [bs,256,21,21]
        lowf = self.conv1(lowf)  # [bs,256,21,21]
        bb,cc,ww,hh = lowf.shape
        highf =self.conv2(highf) # [bs,256,21,21]
        # print("lowf ={} | highf ={}".format(lowf.shape,highf.shape))
        fusion = self.conv3(lowf+highf).flatten(2).transpose(1,0)  # [bs,256,441]
        lowf = lowf.flatten(2).transpose(1,0)      # [bs,256,441]
        query_info= self.multihead(lowf,fusion,fusion)[0].transpose(1,0).view(bb,cc,ww,hh)
        query_info = torch.cat([query_info,lowf.transpose(1,0).view(bb,cc,ww,hh)],dim=1)
        query_info = self.outconv(query_info)
        return self.up_conv(query_info),fusion.transpose(1,0).view(bb,cc,ww,hh)


class SFFNet(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=64):
        super(SFFNet, self).__init__()

        in_ch = 3
        out_ch = 2
        n1 = channel
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
        self.dsppt = DSPP(in_channels=1024,out_ch=512)
        self.dspp1 = DSPP(in_channels=512,out_ch=256)
        self.dspp2 = DSPP(in_channels=256,out_ch=256)
        self.dspp3 = DSPP(in_channels=128,out_ch=128)
        self.mstb1 = MSTB(in_ch=512,out_ch=512,down_ratio=1)
        self.mstb2 = MSTB(in_ch=256,out_ch=256,down_ratio=2)
        self.mstb3 = MSTB(in_ch=256, out_ch=256, down_ratio=4)
        self.fsf1 = FSF(low_ch=512,high_ch=256,up_rate=2)
        self.fsf2 = FSF(low_ch=256,high_ch=256,up_rate=4)
        self.up1 = up_conv(512, 256)
        self.up1_conv = conv_block(512,256)
        self.up2 = up_conv(256, 128)
        self.up2_conv = conv_block(384, 256)
        self.up3 = up_conv(256, 128)
        self.up3_conv = conv_block(256, 128)
        self.up4 = up_conv(128, 64)
        self.outconv = nn.Conv2d(64, 2, 3, padding=1, bias=False)

    def forward(self, x):
        e1 = self.Conv1(x)
        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)         # 128, 168, 168
        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)         # 256, 84, 84
        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)         # 512, 42, 42
        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)         # 512, 21, 21
        e5 = self.dsppt(e5)         # 512,21,21
        e4 = self.dspp1(e4)         # [b, 256, 42,42]
        e3 = self.dspp2(e3)         # [b,256, 84,84]
        e2 = self.dspp3(e2)         # [b,128, 168, 168]

        e5 = self.mstb1(e5)      # [2, 512, 21, 21]
        e4 = self.mstb2(e4)      # [2, 256, 21, 21]
        e3 = self.mstb3(e3)      # [2, 256, 21, 21]

        e4,memory = self.fsf1(e5,e4)  # e4[2, 256, 42, 42]   memory[2, 256, 21, 21]
        e5 = self.up1(e5)             # e5[bs,256,42,42]
        e5 = torch.cat([e4,e5],dim=1) # e5[bs,512,42,42]
        e5 = self.up1_conv(e5)       # [bs,256,42,42]

        e4, _ = self.fsf2(memory, e3) # e4[2, 256, 84, 84]  memory[2, 256, 21, 21]
        e5 = self.up2(e5)   # [bs,128,84,84]
        e5 = torch.cat([e4, e5], dim=1) # [bs,384,84,84]
        e5 = self.up2_conv(e5)      # [bs,256,84,84]
        e5 = self.up3(e5)           # [bs,128,168,168]
        e5 = torch.cat([e5,e2],dim=1)     # [bs,256,168,168]
        e5 = self.up3_conv(e5)
        e5 = self.up4(e5)           # [bs,64,336,336]
        e5 = self.outconv(e5)       # [bs,2,336,336]
        return  e5



"""# input = torch.randn(2,512,441)
# input = torch.randn(2,512,21,21)
input = torch.randn(2,3,336,336)
# model = PraNet()
# model = DSPP(in_channels=512)
model = SFFNet(64)
# model = MSTB(in_ch=512,out_ch=512,down_ratio=1)
out = model(input)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))
print("out shape = ",out.shape)"""

