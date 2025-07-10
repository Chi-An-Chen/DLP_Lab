"""
Author: Chi-An Chen
Date: 2025-07-10
Description:
"""
import torch
import numpy as np
import torch.nn as nn

def dice_score(preds, targets, eps=1e-6):
    preds = preds.float()
    targets = targets.float()
    preds = preds.view(preds.size(0), -1)    # [B, N]
    targets = targets.view(targets.size(0), -1)  # [B, N]
    dice_list = []
    for p, t in zip(preds, targets):
        mask = t > 0
        if mask.sum() == 0:
            dice_list.append(torch.tensor(1.0, device=preds.device))
        else:
            intersection = (p * t)[mask].sum()
            union = p[mask].sum() + t[mask].sum()
            dice_val = (2. * intersection + eps) / (union + eps)
            dice_list.append(dice_val)
    return torch.stack(dice_list).mean()

def compute_metrics(preds, mask, threshold=0.5):
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)  # [B, 1, H, W]

    TP = (preds * mask).sum(dim=(1,2,3))
    FP = (preds * (1 - mask)).sum(dim=(1,2,3))
    FN = ((1 - preds) * mask).sum(dim=(1,2,3))

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    dice = 2 * TP / (2 * TP + FP + FN + 1e-8)
    iou = TP / (TP + FP + FN + 1e-8)

    return (
        precision.mean().item(),
        recall.mean().item(),
        dice.mean().item(),
        iou.mean().item()
    )

def bce_loss_with_logits(logits, targets, eps=1e-7):
    probs = torch.sigmoid(logits)
    probs = torch.clamp(probs, eps, 1.0 - eps)
    loss = -targets * torch.log(probs) - (1 - targets) * torch.log(1 - probs)
    return loss.mean()

def dice_loss_with_logits(logits, targets, eps=1e-7):
    probs = torch.sigmoid(logits)
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + eps) / (union + eps)
    loss = 1.0 - dice
    return loss.mean()

def combined_loss(logits, targets, bce_weight=0.5, dice_weight=0.5):
    bce = bce_loss_with_logits(logits, targets)
    dice = dice_loss_with_logits(logits, targets)
    return bce_weight * bce + dice_weight * dice
