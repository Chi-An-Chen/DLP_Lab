"""
Author: Chi-An Chen
Date: 2025-07-10
Description: Model [Resnet encoder + UNet decoder] with self-design attention
"""
import torch
from torch import nn

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
    

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride,
                                   padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.silu(x)
        return x


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResNetBlock, self).__init__()
        self.block = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            DepthwiseSeparableConv(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
        )
        self.down = None
        if stride!=1 or in_channels != out_channels:
            self.down = nn.Sequential(
                DepthwiseSeparableConv(in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.block(x)
        if self.down is not None:
            identity = self.down(identity)
        x += identity
        x = self.silu(x)
        return x
    
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_blocks):
        super(EncoderBlock, self).__init__()
        self.blocks = [ResNetBlock(in_channels, out_channels, 2)]
        for _ in range(1, n_blocks):
            self.blocks.append(ResNetBlock(out_channels, out_channels, 1))
        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        out = self.blocks(x)
        return out, x 
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = nn.Sequential(
            DepthwiseSeparableConv(in_channels+out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            DepthwiseSeparableConv(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
        self.attn = Attention(out_channels)
        
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        x = self.block(x)
        x = self.attn(x)
        return x

class ResNet34_UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(ResNet34_UNet, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.encoder2 = EncoderBlock(64, 64, 3)
        self.encoder3 = EncoderBlock(64, 128, 4)
        self.encoder4 = EncoderBlock(128, 256, 6)
        self.encoder5 = EncoderBlock(256, 512, 3)
        
        self.center = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.SiLU(inplace=True),
            Attention(512),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.decoder4 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)
        self.decoder1 = DecoderBlock(64, 32)
        
        self.output = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1)
        )
    
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2, _ = self.encoder2(enc1)
        enc3, skip1 = self.encoder3(enc2)
        enc4, skip2 = self.encoder4(enc3)
        enc5, skip3 = self.encoder5(enc4)
        
        skip4 = enc5
        center = self.center(enc5)
        
        dec4 = self.decoder4(center, skip4)
        dec3 = self.decoder3(dec4, skip3)
        dec2 = self.decoder2(dec3, skip2)
        dec1 = self.decoder1(dec2, skip1)
        
        return self.output(dec1)
    
if __name__ == "__main__":
    from thop import profile
    model = ResNet34_UNet()

    x = torch.randn(1, 3, 256, 256)
    out = model(x)
    flops, params = profile(model, inputs=(x,))
    print(f"FLOPs: {flops / 1e9:.4f} GFLOPs")
    print(f"Params: {params / 1e6:.4f} M")