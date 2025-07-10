"""
Author: Chi-An Chen
Date: 2025-07-10
Description: Evaluate model performance
"""
import torch

from tqdm import tqdm
from utils import dice_score, compute_metrics

def evaluate_unet(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    total_dice = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_iou = 0.0
    with torch.no_grad():
        print("Calculating Dice Score on test set...")
        for sample in tqdm(test_loader, desc='Evaluating'):
            images = sample['image'].float().to(device)
            masks = sample['mask'].float().to(device)
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > 0.5).float()
            total_dice += dice_score(outputs, masks)
            
            precision, recall, dice, iou = compute_metrics(outputs, masks)
            total_precision+=precision
            total_recall+=recall
            total_iou+=iou
    
    avg_dice = total_dice / len(test_loader)
    avg_precision = total_precision / len(test_loader)
    avg_recall = total_recall / len(test_loader)
    avg_iou = total_iou / len(test_loader)
    return avg_dice, avg_precision, avg_recall, avg_iou