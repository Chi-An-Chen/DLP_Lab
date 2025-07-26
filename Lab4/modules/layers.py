"""
Author: Chi-An Chen
Date: 2025-07-27
Description: VAE layers in modules
"""
import torch
import torch.nn as nn


class AdaptiveSEBlock(nn.Module):
    def __init__(self, channels):
        super(AdaptiveSEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        if channels <= 4:
            hidden_dim = channels
        elif channels <= 16:
            hidden_dim = max(1, channels // 2)
        else:
            hidden_dim = max(4, channels // 16)
        
        self.excitation = nn.Sequential(
            nn.Linear(channels, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)
        else:
            self.skip = None
        self.se = AdaptiveSEBlock(out_ch)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)
        out = self.se(out)
        
        if self.skip is not None:
            identity = self.skip(x)

        out = out + identity
        return out


class DepthConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, depth_kernel=3, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            DepthConv(in_ch, out_ch, depth_kernel, stride),
            ConvFFN(out_ch),
        )

    def forward(self, x):
        return self.block(x)


class ConvFFN(nn.Module):
    def __init__(self, in_ch, slope=0.1, inplace=False):
        super().__init__()
        expansion_factor = 2
        slope = 0.1
        internal_ch = in_ch * expansion_factor
        self.conv = nn.Conv2d(in_ch, internal_ch * 2, 1)
        self.conv_out = nn.Conv2d(internal_ch, in_ch, 1)
        self.relu = nn.LeakyReLU(negative_slope=slope, inplace=inplace)

    def forward(self, x):
        identity = x
        x1, x2 = self.conv(x).chunk(2, 1)
        out = x1 * self.relu(x2)
        return identity + self.conv_out(out)


class DepthConv(nn.Module):
    def __init__(self, in_ch, out_ch, depth_kernel=3, stride=1, slope=0.01, inplace=False):
        super().__init__()
        dw_ch = in_ch * 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, dw_ch, 1, stride=stride),
            nn.LeakyReLU(negative_slope=slope, inplace=inplace),
        )
        self.depth_conv = nn.Conv2d(dw_ch, dw_ch, depth_kernel, padding=depth_kernel // 2,
                                    groups=dw_ch)
        self.conv2 = nn.Conv2d(dw_ch, out_ch, 1)

        self.adaptor = None
        if stride != 1:
            assert stride == 2
            self.adaptor = nn.Conv2d(in_ch, out_ch, 2, stride=2)
        elif in_ch != out_ch:
            self.adaptor = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        identity = x
        if self.adaptor is not None:
            identity = self.adaptor(identity)

        out = self.conv1(x)
        out = self.depth_conv(out)
        out = self.conv2(out)

        return out + identity