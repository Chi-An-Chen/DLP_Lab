"""
Author: Chi-An Chen
Date: 2025-07-10
Description: Training code
"""

import os
import torch
import argparse
import torch.optim as optim

from tqdm import tqdm
from models.unet import UNet
from oxford_pet import load_dataset
from torch.utils.data import DataLoader
from models.resnet34_unet import ResNet34_UNet
from utils import combined_loss, compute_metrics

def train(args):
    # implement the training function here
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Lists to store metrics for plotting
    train_losses = []
    val_dice_scores = []
    
    print(f'=============== Start Training {args.model.upper()}, Using {device} ===============', end='\n\n')
    if args.model.lower() == 'unet':
        model = UNet(in_channels=3, out_channels=1)
    elif args.model.lower() == 'resnet34_unet':
        model = ResNet34_UNet(in_channels=3, out_channels=1)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    train_dataset = load_dataset(args.data_path, mode="train")
    val_dataset = load_dataset(args.data_path, mode="valid")
    # Train : 3312, Val : 368, Test : 3669
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Training
    best_score = 0.0
    train_losses = []
    val_dice_scores = []
    val_precision_scores = []
    val_recall_scores = []
    val_iou_scores = []

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{args.epochs}') as pbar:
            for batch_idx, sample in enumerate(train_loader):
                images = sample['image'].float().to(device)
                masks = sample['mask'].float().to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = combined_loss(outputs, masks)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix({'loss': loss.item()})
        
        # Calculate average loss for the epoch
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_dice = 0
        val_precision = 0.0
        val_recall = 0.0
        val_dice = 0.0
        val_iou = 0.0
        with torch.no_grad():
            for sample in val_loader:
                images = sample['image'].float().to(device)
                masks = sample['mask'].float().to(device)
                outputs = model(images)
                outputs = torch.sigmoid(outputs)
                outputs = (outputs > 0.5).float()
                precision, recall, dice, iou = compute_metrics(outputs, masks)
                val_precision += precision
                val_recall   += recall
                val_dice     += dice
                val_iou      += iou
    
        
        avg_val_dice      = val_dice / len(val_loader)
        avg_val_precision = val_precision / len(val_loader)
        avg_val_recall    = val_recall / len(val_loader)
        avg_val_iou       = val_iou / len(val_loader)

        val_dice_scores.append(avg_val_dice)
        val_precision_scores.append(avg_val_precision)
        val_recall_scores.append(avg_val_recall)
        val_iou_scores.append(avg_val_iou)
        
        print(f'Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Dice = {avg_val_dice:.4f}, Precision = {avg_val_precision:.4f}, Recall = {avg_val_recall:.4f}, IoU = {avg_val_iou:.4f}')
        
        # Save model if validation dice score improves
        if avg_val_dice > best_score:
            best_score = avg_val_dice
            os.makedirs(args.save_dir, exist_ok=True)
            model_path = f'best_model.pth'
            torch.save(model.state_dict(), os.path.join(args.save_dir, model_path))
        else:
            os.makedirs(args.save_dir, exist_ok=True)
            model_path = f'last.pth'
            torch.save(model.state_dict(), os.path.join(args.save_dir, model_path))

    txt_path = os.path.join(args.save_dir, 'output.txt')
    with open(txt_path, 'w') as f:
        for i in range(args.epochs):
            text = f'[{train_losses[i]}, {val_dice_scores[i]}, {val_precision_scores[i]}, {val_recall_scores[i]}, {val_iou_scores[i]}]'
            f.write(f"{text}\n")

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path'    , type=str, default='dataset', help='path of the input data')
    parser.add_argument('--model'        , type=str, default='unet', choices=['unet', 'resnet34_unet'], help='model architecture to use')
    parser.add_argument('--epochs'       , '-e', type=int, default=200, help='number of epochs')
    parser.add_argument('--batch_size'   , '-b', type=int, default=16, help='batch size')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--use_scheduler', action='store_true', help='use learning rate scheduler')
    parser.add_argument('--load_epoch'   , type=int, default=0, help='load model from specific epoch')
    parser.add_argument('--save_interval', type=int, default=10, help='save model every N epochs')
    parser.add_argument('--save_dir'     , type=str, default='saved_models_resnet34_unet', help='directory to save models and plots')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    train(args)