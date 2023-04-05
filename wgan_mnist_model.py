import torch
import torch.nn.functional as F
from torch import nn


class DownConv(nn.Module):
    """Downscaling block with convolution."""
    def __init__(self, input_size, output_size, kernel_size=5, stride=1, padding=0, neg_slope=0.01):
        super().__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(input_size, output_size, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(neg_slope)
        )

    def forward(self, x):
        return self.convblock(x)


class UpConv(nn.Module):
    """Upscaling block with convolution transpose."""
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=0, output_padding=0):
        super().__init__()
        self.convblock = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.convblock(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.dc1 = DownConv(1, 128, kernel_size=5, stride=2, padding=2, neg_slope=0.1)  # 128*14*14
        self.dc2 = DownConv(128, 256, kernel_size=5, stride=2, padding=2, neg_slope=0.1)  # 256*7*7
        self.dc3 = DownConv(256, 512, kernel_size=5, stride=2, padding=2, neg_slope=0.1)  # 512*4*4
        self.out = nn.Linear(512*4*4, 1)

    def forward(self, x):
        x = self.dc1(x)
        # print(x.shape)
        x = self.dc2(x)
        # print(x.shape)
        x = self.dc3(x)
        # print('after downconv blocks', x.shape)
        out = self.out(torch.flatten(x, 1))
        return out.view(-1, 1)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 512*4*4)  # 512*4*4
        self.uc1 = UpConv(512, 256, kernel_size=5, stride=1, padding=0)  # 256*8*8
        self.uc2 = UpConv(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1)  # 128*16*16
        self.conv1 = nn.ConvTranspose2d(128, 1, kernel_size=5, stride=2, padding=2, output_padding=1)  # 1*32*32
        self.down = nn.Upsample(size=(28, 28), mode='bilinear')  # 1*28*28

    def forward(self, x):
        x = self.fc1(x)
        x = self.uc1(x.view(-1, 512, 4, 4))
        # print('after upconv block1', x.shape)
        x = self.uc2(x)
        # print('after upconv block2', x.shape)
        x = F.sigmoid(self.conv1(x))
        # print('after conv-sigmoid', x.shape)
        out = self.down(x)
        return out