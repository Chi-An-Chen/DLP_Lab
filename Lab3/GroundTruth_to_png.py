"""
Author: Chi-An Chen
Date: 2025-07-18
Description: Turn ground truth from csv to images
"""

import os
import torch
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

df = pd.read_csv('faster-pytorch-fid/test_gt.csv')
print(f"Total images: {len(df)}")

save_dir = 'Result/ground_truth'
os.makedirs(save_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Saving images"):
    
    flattened_img_array = np.array(row) / 255.0
    img = flattened_img_array.reshape((64, 64, 3))

    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    save_path = os.path.join(save_dir, f'{idx:04d}.png')
    pil_img.save(save_path)

print(f"All images saved to: {save_dir}")