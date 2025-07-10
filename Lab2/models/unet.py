"""
Author: Chi-An Chen
Date: 2025-07-10
Description: Model [UNet Architecture] with self-design attention
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, c2, attn_bias=False, proj_drop=0.):
        super().__init__()
        self.c2 = c2
        self.qkv = nn.Conv2d(c2, 3 * c2, 1, stride=1, padding=0, bias=attn_bias)

        self.spatial_conv       = nn.Conv2d(c2, c2, 3, 1, 1, groups=c2)
        self.spatial_bn         = nn.BatchNorm2d(c2)
        self.spatial_activation = nn.SELU()
        self.spatial_out        = nn.Conv2d(c2, 1, 1, 1, 0, bias=False)

        self.channel_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.channel_fc   = nn.Conv2d(c2, c2, 1, 1, 0, bias=False)

        self.dwc  = nn.Conv2d(c2, c2, 3, 1, 1, groups=c2)
        self.proj = nn.Conv2d(c2, c2, 3, 1, 1, groups=c2)
        self.proj_drop = nn.Dropout(proj_drop)

    def spatial_channel_attention(self, x):
        # Spatial Operation
        s = self.spatial_conv(x)
        s = self.spatial_bn(s)
        s = self.spatial_activation(s)
        s = self.spatial_out(s).sigmoid()

        # Channel Operation
        c = self.channel_pool(x)
        c = self.channel_fc(c).sigmoid()

        return x * s * c
    
    def forward(self, x):
        # Calculate Q, K, V
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q = self.spatial_channel_attention(q)
        k = self.spatial_channel_attention(k)
        out = self.proj(self.dwc(q + k) * v)
        out = self.proj_drop(out)
        return out


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.dwconv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),  # Depthwise
            nn.Conv2d(in_channels, out_channels, kernel_size=1),  # Pointwise
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )
        
        self.dwconv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels),  # Depthwise
            nn.Conv2d(out_channels, out_channels, kernel_size=1),  # Pointwise
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Dropout(p=0.2)
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.dwconv1(x)
        x = self.dwconv2(x)
        return self.pool(x)
    
class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.dwconv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),  # Depthwise
            nn.Conv2d(in_channels, out_channels, kernel_size=1),  # Pointwise
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )
        
        self.dwconv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels),  # Depthwise
            nn.Conv2d(out_channels, out_channels, kernel_size=1),  # Pointwise
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Dropout(p=0.2)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.dwconv1(x)
        x = self.dwconv2(x)
        return self.pool(x)


class DecoderBlock(nn.Module):
    def __init__(self, up_in_channels, enc_in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(up_in_channels, enc_in_channels, kernel_size=2, stride=2)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(enc_in_channels * 2, out_channels, kernel_size=3, padding=1, groups=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Dropout(p=0.2)
        )

        self.attention = Attention(out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.attention(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        self.encoder1 = EncoderBlock(in_channels, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = BottleneckBlock(512, 1024)
        self.bottleneck_attn = Attention(1024)

        self.decoder4 = DecoderBlock(1024, 512, 512)
        self.decoder3 = DecoderBlock(512, 256, 256)
        self.decoder2 = DecoderBlock(256, 128, 128)
        self.decoder1 = DecoderBlock(128, 64, 64)

        self.output_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        bottleneck = self.bottleneck(enc4)

        dec4 = self.decoder4(bottleneck, enc4)
        dec3 = self.decoder3(dec4, enc3)
        dec2 = self.decoder2(dec3, enc2)
        dec1 = self.decoder1(dec2, enc1)
        out  = self.output_conv(dec1)
        return F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
 
if __name__ == "__main__":
    from thop import profile
    model = UNet()

    x = torch.randn(1, 3, 256, 256)
    out = model(x)
    flops, params = profile(model, inputs=(x,))
    print(f"FLOPs: {flops / 1e9:.4f} GFLOPs")
    print(f"Params: {params / 1e6:.4f} M")