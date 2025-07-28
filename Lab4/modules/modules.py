"""
Author: Chi-An Chen
Date: 2025-07-27
Description: VAE modules
"""
import torch
import torch.nn as nn

from torch.autograd import Variable
from .layers import DepthConvBlock, EnhancedResidualBlock, SafeGroupNorm

__all__ = [
    "ProgressiveGenerator",
    "EnhancedRGB_Encoder",
    "EnhancedGaussianPredictor",
    "EnhancedDecoderFusion",
    "EnhancedLabel_Encoder",
    "MultiScaleEncoder"
]


class ProgressiveGenerator(nn.Module):
    def __init__(self, input_nc, output_nc):
        super().__init__()
        self.stages = nn.ModuleList([
            # Stage 1: 低分辨率
            nn.Sequential(
                DepthConvBlock(input_nc, input_nc),
                EnhancedResidualBlock(input_nc, input_nc//2),
            ),
            # Stage 2: 中分辨率
            nn.Sequential(
                DepthConvBlock(input_nc//2, input_nc//2),
                EnhancedResidualBlock(input_nc//2, input_nc//4),
            ),
            # Stage 3: 高分辨率
            nn.Sequential(
                DepthConvBlock(input_nc//4, input_nc//4),
                EnhancedResidualBlock(input_nc//4, input_nc//8),
            ),
            # Stage 4: 最終輸出
            nn.Sequential(
                DepthConvBlock(input_nc//8, input_nc//8),
                nn.Conv2d(input_nc//8, 3, 1),
                nn.Sigmoid()
            )
        ])
        
        self.current_stage = len(self.stages) - 1
        self.alpha = 1.0
        
    def forward(self, input, stage=None):
        if stage is None:
            stage = self.current_stage
            
        x = input
        for i, stage_module in enumerate(self.stages):
            if i <= stage:
                x = stage_module(x)
            else:
                break
                
        return x
    
    def grow_network(self):
        """逐漸增加網絡複雜度"""
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            self.alpha = 0.0
    
    def update_alpha(self, progress):
        """更新alpha用於smooth transition"""
        self.alpha = min(1.0, progress)


class MultiScaleEncoder(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.scales = nn.ModuleList([
            nn.Conv2d(in_chans, out_chans//4, 3, padding=1),
            nn.Conv2d(in_chans, out_chans//4, 5, padding=2),
            nn.Conv2d(in_chans, out_chans//4, 7, padding=3),
            nn.Conv2d(in_chans, out_chans//4, 1)
        ])
        
    def forward(self, x):
        features = [scale(x) for scale in self.scales]
        return torch.cat(features, dim=1)


class EnhancedRGB_Encoder(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.multi_scale = MultiScaleEncoder(in_chans, out_chans//4)
        
        self.encoder_layers = nn.Sequential(
            EnhancedResidualBlock(out_chans//4, out_chans//8),
            DepthConvBlock(out_chans//8, out_chans//8),
            EnhancedResidualBlock(out_chans//8, out_chans//4),
            DepthConvBlock(out_chans//4, out_chans//4),
            EnhancedResidualBlock(out_chans//4, out_chans//2),
            DepthConvBlock(out_chans//2, out_chans//2),
            nn.Conv2d(out_chans//2, out_chans, 3, padding=1),
        )
        
    def forward(self, image):
        multi_scale_features = self.multi_scale(image)
        return self.encoder_layers(multi_scale_features)

class EnhancedLabel_Encoder(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_chans, out_chans//2, kernel_size=7, padding=0),
            SafeGroupNorm(8, out_chans//2),  # 使用SafeGroupNorm
            nn.GELU(),
            EnhancedResidualBlock(in_ch=out_chans//2, out_ch=out_chans)
        )
        
    def forward(self, image):
        return self.encoder(image)


class EnhancedGaussianPredictor(nn.Module):
    def __init__(self, in_chans=48, out_chans=96):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            EnhancedResidualBlock(in_chans, out_chans//4),
            DepthConvBlock(out_chans//4, out_chans//4),
            EnhancedResidualBlock(out_chans//4, out_chans//2),
            DepthConvBlock(out_chans//2, out_chans//2),
            EnhancedResidualBlock(out_chans//2, out_chans),
        )
        
        self.mu_predictor = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.GELU(),
            nn.Conv2d(out_chans, out_chans, kernel_size=1)
        )
        
        self.logvar_predictor = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.GELU(),
            nn.Conv2d(out_chans, out_chans, kernel_size=1)
        )
        
        self.register_buffer('beta', torch.tensor(1.0))
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def kl_divergence(self, mu, logvar):
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1,2,3])
        return self.beta * kl.mean()
    
    def forward(self, img, label):
        feature = torch.cat([img, label], dim=1)
        encoded = self.feature_extractor(feature)
        
        mu = self.mu_predictor(encoded)
        logvar = self.logvar_predictor(encoded)
        
        z = self.reparameterize(mu, logvar)
        kl_loss = self.kl_divergence(mu, logvar)
        
        return z, mu, logvar, kl_loss
    
    def update_beta(self, epoch, max_epochs, min_beta=0.0, max_beta=1.0):
        self.beta = min_beta + (max_beta - min_beta) * min(epoch / (max_epochs * 0.5), 1.0)


class EnhancedDecoderFusion(nn.Module):
    def __init__(self, img_channels, label_channels, z_channels, out_channels):
        super().__init__()
        
        self.img_proj = nn.Conv2d(img_channels, img_channels, 1)
        self.label_proj = nn.Conv2d(label_channels, label_channels, 1)
        self.z_proj = nn.Conv2d(z_channels, z_channels, 1)
        
        total_channels = img_channels + label_channels + z_channels
        
        if total_channels > 256:
            self.channel_reduce = nn.Conv2d(total_channels, 256, 1)
            fusion_input_channels = 256
        else:
            self.channel_reduce = None
            fusion_input_channels = total_channels
        
        self.fusion_layers = nn.Sequential(
            DepthConvBlock(fusion_input_channels, fusion_input_channels),
            EnhancedResidualBlock(fusion_input_channels, fusion_input_channels//4),
            DepthConvBlock(fusion_input_channels//4, fusion_input_channels//2),
            EnhancedResidualBlock(fusion_input_channels//2, fusion_input_channels//2),
            DepthConvBlock(fusion_input_channels//2, out_channels//2),
            nn.Conv2d(out_channels//2, out_channels, 1, 1)
        )
        
    def forward(self, img, label, z):
        img_feat = self.img_proj(img)
        label_feat = self.label_proj(label)
        z_feat = self.z_proj(z)
        
        fused_feature = torch.cat([img_feat, label_feat, z_feat], dim=1)
        
        if self.channel_reduce is not None:
            fused_feature = self.channel_reduce(fused_feature)
        
        return self.fusion_layers(fused_feature)


class CrossModalAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.query = nn.Conv2d(channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, img, label, z):
        # 將三個特徵連接
        combined = torch.cat([img, label, z], dim=1)
        B, C, H, W = combined.shape
        
        # 計算注意力
        q = self.query(combined).view(B, self.channels, -1)
        k = self.key(combined).view(B, self.channels, -1)
        v = self.value(combined).view(B, self.channels, -1)
        
        attention = self.softmax(torch.bmm(q.transpose(1, 2), k))
        out = torch.bmm(v, attention.transpose(1, 2))
        
        return out.view(B, self.channels, H, W)

class ProgressiveGenerator(nn.Module):
    def __init__(self, input_nc, output_nc):
        super().__init__()
        self.stages = nn.ModuleList([
            # Stage 1: 低分辨率
            nn.Sequential(
                DepthConvBlock(input_nc, input_nc),
                EnhancedResidualBlock(input_nc, input_nc//2),
            ),
            # Stage 2: 中分辨率
            nn.Sequential(
                DepthConvBlock(input_nc//2, input_nc//2),
                EnhancedResidualBlock(input_nc//2, input_nc//4),
            ),
            # Stage 3: 高分辨率
            nn.Sequential(
                DepthConvBlock(input_nc//4, input_nc//4),
                EnhancedResidualBlock(input_nc//4, input_nc//8),
            ),
            # Stage 4: 最終輸出
            nn.Sequential(
                DepthConvBlock(input_nc//8, input_nc//8),
                nn.Conv2d(input_nc//8, 3, 1),
                nn.Sigmoid()
            )
        ])
        
        self.current_stage = len(self.stages) - 1
        self.alpha = 1.0  # 用於smooth transition
        
    def forward(self, input, stage=None):
        if stage is None:
            stage = self.current_stage
            
        x = input
        for i, stage_module in enumerate(self.stages):
            if i <= stage:
                x = stage_module(x)
            else:
                break
                
        return x
    
    def grow_network(self):
        """逐漸增加網絡複雜度"""
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            self.alpha = 0.0  # 重置alpha進行smooth transition
    
    def update_alpha(self, progress):
        """更新alpha用於smooth transition"""
        self.alpha = min(1.0, progress)
