import pandas as pd
import numpy as np
import pandas as pd
import os
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# 讀取 CSV
df = pd.read_csv('/Users/anguschen/Downloads/lab3/faster-pytorch-fid/test_gt.csv')
print(f"Total images: {len(df)}")

# 建立 ground_truth 資料夾
save_dir = '/Users/anguschen/Downloads/lab3/Result/ground_truth'
os.makedirs(save_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 儲存圖片（加上 tqdm 進度條）
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Saving images"):
    
    flattened_img_array = np.array(row) / 255.0  # 正規化到 0~1
    img = flattened_img_array.reshape((64, 64, 3))  # 還原成圖片格式

    pil_img = Image.fromarray((img * 255).astype(np.uint8))  # 回轉為 uint8
    save_path = os.path.join(save_dir, f'{idx:04d}.png')
    pil_img.save(save_path)

print(f"✅ All images saved to: {save_dir}")