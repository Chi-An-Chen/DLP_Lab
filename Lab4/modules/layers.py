"""
Author: Chi-An Chen
Date: 2025-07-27
Description: VAE layers in modules
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
        
    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        attention = torch.cat([avg_pool, max_pool], dim=1)
        attention = torch.sigmoid(self.conv(attention))
        return x * attention


class SafeGroupNorm(nn.Module):
    """安全的GroupNorm，自動處理通道數不整除的情況"""
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__()
        
        for groups in range(min(num_groups, num_channels), 0, -1):
            if num_channels % groups == 0:
                self.norm = nn.GroupNorm(groups, num_channels, eps, affine)
                break
        else:
            print(f"Warning: Using BatchNorm instead of GroupNorm for {num_channels} channels")
            self.norm = nn.BatchNorm2d(num_channels, eps=eps, affine=affine)
    
    def forward(self, x):
        return self.norm(x)
class EnhancedResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.bn1 = SafeGroupNorm(8, out_ch)  # 使用SafeGroupNorm
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn2 = SafeGroupNorm(8, out_ch)
        
        if in_ch != out_ch or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride),
                SafeGroupNorm(8, out_ch)
            )
        else:
            self.skip = None
            
        self.se = AdaptiveSEBlock(out_ch)
            
    def forward(self, x):
        identity = x
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out = self.se(out)
        
        if self.skip is not None:
            identity = self.skip(x)
        return F.gelu(out + identity)


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
