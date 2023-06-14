import torch.nn as nn
import torch
from torch import autograd
from net.qau_attention import *


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(),
        )

    def forward(self, input):
        return self.conv(input)


class object_dp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(object_dp, self).__init__()
        self.tconv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
            nn.PReLU(),
        )

    def forward(self, input):
        return self.tconv(input)


class before_sp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(before_sp, self).__init__()
        self.bconv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(),
        )

    def forward(self, input):
        return self.bconv(input)


class QAU_Net(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(QAU_Net, self).__init__()

        self.conv1 = DoubleConv(in_ch, 64)
        self.before_sp1 = before_sp(128, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.object_dp1 = object_dp(64, 64)
        self.conv2 = DoubleConv(64, 128)
        self.before_sp2 = before_sp(256, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.object_dp2 = object_dp(128, 128)
        self.conv3 = DoubleConv(128, 256)
        self.before_sp3 = before_sp(512, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.object_dp3 = object_dp(256, 256)
        self.conv4 = DoubleConv(256, 512)
        self.before_sp4 = before_sp(1024, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.object_dp4 = object_dp(512, 512)
        self.conv5 = DoubleConv(512, 1024)

        self.quartet_attention = QuartetAttention(1024, 16)

        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_ch, 1)
        # self.sm = torch.nn.Softmax(dim=1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        obdp1 = self.object_dp1(p1)
        cat1 = torch.cat([c1, obdp1], dim=1)
        bsp1 = self.before_sp1(cat1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        obdp2 = self.object_dp2(p2)
        cat2 = torch.cat([c2, obdp2], dim=1)
        bsp2 = self.before_sp2(cat2)

        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        obdp3 = self.object_dp3(p3)
        cat3 = torch.cat([c3, obdp3], dim=1)
        bsp3 = self.before_sp3(cat3)

        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        obdp4 = self.object_dp4(p4)
        cat4 = torch.cat([c4, obdp4], dim=1)
        bsp4 = self.before_sp4(cat4)

        c5 = self.conv5(p4)
        c5 = self.quartet_attention(c5)
        # print("c5 shape = ",c5.shape)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, bsp4], dim=1)
        c6 = self.conv6(merge6)
        # print("c6 shape = ", c6.shape)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, bsp3], dim=1)
        c7 = self.conv7(merge7)
        # print("c7 shape = ", c7.shape)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, bsp2], dim=1)
        c8 = self.conv8(merge8)
        # print("c8 shape = ", c8.shape)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, bsp1], dim=1)
        c9 = self.conv9(merge9)
        # print("c9 shape = ", c9.shape)
        out = self.conv10(c9)
        # out = self.sm(c10)
        return out


"""if __name__ == "__main__":
    model = QAU_Net(3, 2)
    input = torch.randn(1, 3, 336, 336)
    output = model(input)
    print(output.shape)"""
