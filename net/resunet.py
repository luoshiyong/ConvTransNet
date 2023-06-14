import torch
import torch.nn as nn
class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)



class ResUnet(nn.Module):
    def __init__(self, channel, filters=[64, 128, 256, 512,1024]):
        super(ResUnet, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.bridge = ResidualConv(filters[2], filters[3], 2, 1)
        self.bridge2 = ResidualConv(filters[3], filters[4], 2, 1)
        self.upsample_0 = Upsample(filters[4], filters[4], 2, 2)

        self.upsample_1 = Upsample(filters[4]+filters[3], filters[3], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], 2, 1, 1),
            #nn.Sigmoid(),
        )

    def forward(self, x):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)# [2, 64, 336, 336]
        x2 = self.residual_conv_1(x1)    # [2, 128, 168, 168]
        x3 = self.residual_conv_2(x2)    # [2, 256, 84, 84]
        # Bridge
        x4 = self.bridge(x3)             # [2, 512, 42, 42]
        x44 = self.bridge2(x4)           # [2, 1024, 21, 21]
        # Decode
        x44 = self.upsample_0(x44)       # [2,1024,21,21]
        x44 = torch.cat([x44,x4],dim=1)  # [2,1536,42,42]
        x4 = self.upsample_1(x44)         # [2, 512, 84, 84]
        x5 = torch.cat([x4, x3], dim=1)  # [2, 768, 84, 84]
        x6 = self.up_residual_conv1(x5)  # [2, 256, 84, 84]
        x6 = self.upsample_2(x6)         # [2, 256, 168, 168]
        x7 = torch.cat([x6, x2], dim=1)  # [2, 384, 168, 168]
        x8 = self.up_residual_conv2(x7)  # [2, 128, 168, 168]
        x8 = self.upsample_3(x8)         # [2, 128, 336, 336]
        x9 = torch.cat([x8, x1], dim=1)  # [2, 192, 336, 336]
        x10 = self.up_residual_conv3(x9) # [2, 64, 336, 336]
        output = self.output_layer(x10)

        return output
"""input = torch.randn(2,3,336,336)
model = ResUnet(3)
out = model(input)
from utilities.utils import  count_params
print(count_params(model))
print(out.shape)"""