"""
Author: Chi-An Chen
Date: 2025-07-10
Description: Inference code
"""
import os
import cv2
import torch
import argparse
import numpy as np

from tqdm import tqdm
from PIL import Image
from models.unet import UNet
from evaluate import evaluate_unet
from oxford_pet import load_dataset
from torch.utils.data import DataLoader
from models.resnet34_unet import ResNet34_UNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def preprocess_data(img_path):
    data = Image.open(img_path).convert("RGB")
    data = np.array(data.resize((256, 256), Image.BILINEAR))
    data = data.astype(np.float32) / 255.0  # 歸一化到 0-1
    data = torch.tensor(data, dtype=torch.float32)
    data = torch.permute(data, (2, 0, 1))
    return data
        
def to_img(data, mask):
    data = data.squeeze(0).cpu().numpy().transpose((1, 2, 0))
    data = (data * 255).astype('uint8')
    if isinstance(mask, np.ndarray) == False:
        mask = mask.squeeze(0).cpu().numpy()
    data[mask == 0] = [0, 0, 0]
    return Image.fromarray(data, 'RGB')

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model'     , type=str, default='unet', choices=['unet', 'resnet34_unet'], help='model architecture to use')
    parser.add_argument('--data_path' , type=str, default='dataset', help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--model_path', type=str, default='saved_models_unet/best_model.pth', help='path to the model weights')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    
    if args.model.lower() == 'unet':
        print("Using UNet model")
        model = UNet(3, 1).to(device)
    elif args.model.lower() == 'resnet34_unet':
        print("Using ResNet34_UNet model")
        model = ResNet34_UNet(3, 1).to(device)
        
    # Loading Weights
    print(f"Loading model from {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    test_loader = DataLoader(load_dataset(args.data_path, "test"), batch_size=args.batch_size, shuffle=False)
    avg_dice, avg_precision, avg_recall, avg_iou = evaluate_unet(model, test_loader)
    print('='*30)
    print(f"Dice Score: {avg_dice * 100:.4f}%")
    print(f"Precision : {avg_precision * 100:.4f}%")
    print(f"Recall    : {avg_recall * 100:.4f}%")
    print(f"IoU       : {avg_iou * 100:.4f}%")
    print('='*30, end='\n\n')
    
    txt_path = f'Result_{args.model}.txt'
    with open(txt_path, 'w') as f:
        f.write(f"Dice Score: {avg_dice:.8f}\n")
        f.write(f"Precision : {avg_precision:.8f}\n")
        f.write(f"Recall    : {avg_recall:.8f}\n")
        f.write(f"Iou       : {avg_iou:.8f}\n")
    
    list_path = os.path.join(args.data_path, 'annotations/test.txt')
    with open(list_path) as f:
        filenames = f.read().strip('\n').split('\n')
    filenames = [x.split(' ')[0] for x in filenames]

    os.makedirs('outputs_imgs', exist_ok=True)
    os.makedirs(os.path.join('outputs_imgs', args.model), exist_ok=True)
    
    print(f'Inferencing ...')
    os.makedirs(os.path.join('outputs_imgs', args.model), exist_ok=True)
    with torch.no_grad():
        for file in tqdm(filenames):
            img_path = os.path.join(args.data_path, 'images', file + '.jpg')
            data = preprocess_data(img_path)
            data = data.unsqueeze(0).to(device)
            mask = model(data)
            mask = (mask > 0).cpu().numpy().squeeze()
            new_img = to_img(data, mask)
            new_img.save(os.path.join('outputs_imgs', args.model, f'{file}_mask.png'), format='PNG')