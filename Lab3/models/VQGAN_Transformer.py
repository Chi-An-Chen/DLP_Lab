"""
Author: Chi-An Chen
Date: 2025-07-17
Description: MaskGit model with VQGAN and BidirectionalTransformer (With self-designed Muti-head attention)
"""

import os
import math
import yaml
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer

class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])

        self.num_image_tokens = configs['num_image_tokens']
        self.mask_token_id = configs['num_codebook_vectors']
        self.choice_temperature = configs['choice_temperature']
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])

        # Register the mask token as a constant buffer (non-trainable, moves with .to(device))
        self.register_buffer('mask_token_tensor', torch.tensor(self.mask_token_id))

    def load_transformer_checkpoint(self, load_ckpt_path):
        self.transformer.load_state_dict(torch.load(load_ckpt_path))

    @staticmethod
    def load_vqgan(configs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param']).to(device)
        state_dict = torch.load(configs['VQ_CKPT_path'], map_location=device)
        model.load_state_dict(state_dict, strict=True)
        return model.eval()

    @torch.no_grad()
    def encode_to_z(self, x):
        zq, z_ind, _ = self.vqgan.encode(x)
        return zq, z_ind

    def gamma_func(self, mode="cosine"):
        """Returns a masking schedule function gamma(ratio) ∈ (0, 1]."""
        if mode == "linear":
            return lambda r: 1 - r
        elif mode == "cosine":
            return lambda r: math.cos(math.pi * r / 2)
        elif mode == "square":
            return lambda r: 1 - r ** 2
        else:
            raise NotImplementedError(f"Unknown gamma_type: {mode}")

    def forward(self, x, ratio):
        _, z_indices = self.encode_to_z(x)
        z_indices = z_indices.view(-1, self.num_image_tokens)

        # Apply random mask
        mask = torch.bernoulli(torch.ones_like(z_indices, dtype=torch.float32) * ratio)
        z_input = torch.where(mask.bool(), self.mask_token_tensor, z_indices)

        # Transformer prediction
        logits = self.transformer(z_input)[..., :self.mask_token_id]

        # One-hot ground truth
        ground_truth = F.one_hot(z_indices, num_classes=self.mask_token_id).float()
        return logits, ground_truth

    @torch.no_grad()
    def inpainting(self, x, ratio, mask_b):
        _, z_indices = self.encode_to_z(x)
        z_indices = z_indices.view(-1, self.num_image_tokens)

        # Apply initial mask
        z_input = torch.where(mask_b.bool(), self.mask_token_tensor, z_indices)

        logits = F.softmax(self.transformer(z_input), dim=-1)
        pred_probs, pred_indices = torch.max(logits, dim=-1)

        # Apply gumbel noise to confidence
        noise = -torch.log(-torch.log(torch.rand_like(pred_probs)))
        gamma_ratio = self.gamma(ratio)
        temperature = self.choice_temperature * (1 - gamma_ratio)
        confidence = pred_probs + temperature * noise

        # Suppress confidence for unmasked tokens
        confidence = torch.where(mask_b.bool(), confidence, torch.full_like(confidence, float('inf')))

        # Select tokens to update
        num_updates = max(1, math.ceil(mask_b.sum().item() * gamma_ratio))
        _, idx_to_update = torch.topk(confidence, num_updates, largest=False)

        # Create updated mask
        mask_update = torch.zeros_like(mask_b).scatter_(1, idx_to_update, 1)
        mask_update = mask_update & mask_b  # keep only original masked positions

        return pred_indices, mask_update

__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}