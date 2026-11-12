import torch
from torch import nn


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """U-Net for binary fetal-head-region segmentation on HC18-style images."""

    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.e1 = conv_block(1, 32)
        self.e2 = conv_block(32, 64)
        self.e3 = conv_block(64, 128)
        self.e4 = conv_block(128, 256)
        self.mid = conv_block(256, 512)
        self.u4 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.d4 = conv_block(512, 256)
        self.u3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.d3 = conv_block(256, 128)
        self.u2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.d2 = conv_block(128, 64)
        self.u1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.d1 = conv_block(64, 32)
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(self.pool(a))
        c = self.e3(self.pool(b))
        d = self.e4(self.pool(c))
        z = self.mid(self.pool(d))
        z = self.d4(torch.cat([self.u4(z), d], dim=1))
        z = self.d3(torch.cat([self.u3(z), c], dim=1))
        z = self.d2(torch.cat([self.u2(z), b], dim=1))
        z = self.d1(torch.cat([self.u1(z), a], dim=1))
        return self.out(z)
