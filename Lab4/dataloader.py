"""
Author: Chi-An Chen
Date: 2025-07-27
Description: Dataloader
"""
import os
import torch

from glob import glob
from torch import stack
from torch.utils.data import Dataset as torchData
from torchvision.datasets.folder import default_loader as imgloader

def get_key(fp):
    filename = os.path.basename(fp)
    filename = filename.split('.')[0].replace('frame', '')
    return filename


class Dataset_Dance(torchData):
    """
        Args:
            root (str)      : The path of your Dataset
            transform       : Transformation to your dataset
            mode (str)      : train, val, test
            partial (float) : Percentage of your Dataset, may set to use part of the dataset
    """
    def __init__(self, root, transform, mode='train', video_len=7, partial=1.0):
        super().__init__()
        assert mode in ['train', 'val'], "There is no such mode !!!"
        if mode == 'train':
            self.img_folder     = sorted(glob(os.path.join(root, 'train/train_img/*.png')), key=get_key)
            self.prefix = 'train'
        elif mode == 'val':
            self.img_folder     = sorted(glob(os.path.join(root, 'val/val_img/*.png')), key=get_key)
            self.prefix = 'val'
        else:
            raise NotImplementedError
        
        self.transform = transform
        self.partial = partial
        self.video_len = video_len

    def __len__(self):
        return int(len(self.img_folder) * self.partial) // self.video_len

    def __getitem__(self, index):
        imgs = []
        labels = []
        for i in range(self.video_len):
            img_name = self.img_folder[(index * self.video_len) + i]
            
            # 使用 os.path 處理路徑，避免 Windows/Linux 路徑分隔符問題
            img_dir = os.path.dirname(img_name)
            parent_dir = os.path.dirname(img_dir)
            filename = os.path.basename(img_name)
            
            label_dir = os.path.join(parent_dir, self.prefix + '_label')
            label_name = os.path.join(label_dir, filename)
            
            # print(f"Loading {img_name} and {label_name}")
            try:
                imgs.append(self.transform(imgloader(img_name)))
                labels.append(self.transform(imgloader(label_name)))
            except FileNotFoundError as e:
                print(f"File not found: {e}")
                raise
                
        return stack(imgs), stack(labels)
