"""
Author: Chi-An Chen
Date: 2025-07-17
Description: Inpainting
"""

import os
import yaml
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision import utils as vutils
from torch.utils.data import Dataset, DataLoader

from utils import LoadTestData, LoadMaskData
from models import MaskGit as VQGANTransformer


class MaskGIT:
    def __init__(self, args, MaskGit_CONFIGS):
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        self.model.load_transformer_checkpoint(args.load_transformer_ckpt_path)
        self.model.eval()
        self.total_iter=args.total_iter
        self.mask_func=args.mask_func
        self.sweet_spot=args.sweet_spot
        self.device=args.device
        self.prepare()

    @staticmethod
    def prepare():
        os.makedirs("./Result/test_results", exist_ok=True)
        os.makedirs("./Result/mask_scheduling", exist_ok=True)
        os.makedirs("./Result/imga", exist_ok=True)

##TODO3 step1-1: total iteration decoding  
#mask_b: iteration decoding initial mask, where mask_b is true means mask
    def inpainting(self, image, mask_b, idx):  
        mean = torch.tensor([0.4868, 0.4341, 0.3844], device=self.device).view(3, 1, 1)
        std = torch.tensor([0.2620, 0.2527, 0.2543], device=self.device).view(3, 1, 1)
        
        imga = torch.zeros(self.total_iter + 1, 3, 64, 64)
        maska = torch.zeros(self.total_iter, 3, 16, 16)

        ori_image = (image[0] * std) + mean
        imga[0] = ori_image

        mask_b = mask_b.to(self.device)
        current_mask = mask_b.clone()

        self.model.eval()
        with torch.no_grad():
            for step in range(self.total_iter):
                if step == self.sweet_spot:
                    break

                ratio = (step + 1) / self.total_iter
                z_indices_pred, current_mask = self.model.inpainting(image, ratio, current_mask)

                # Save current mask visualization
                mask_i = current_mask.view(1, 16, 16)
                mask_image = torch.ones(3, 16, 16)
                indices = torch.nonzero(mask_i, as_tuple=False)
                mask_image[:, indices[:, 1], indices[:, 2]] = 0
                maska[step] = mask_image

                # Decode latent code into image
                z_q = self.model.vqgan.codebook.embedding(z_indices_pred).view(1, 16, 16, 256).permute(0, 3, 1, 2)
                decoded_img = self.model.vqgan.decode(z_q)
                image = decoded_img  # next iteration uses this

                dec_img_ori = (decoded_img[0] * std) + mean
                imga[step + 1] = dec_img_ori

            # Save final output
            vutils.save_image(dec_img_ori, f"./Result/test_results/image_{idx:03d}.png", nrow=1)
            vutils.save_image(maska, f"./Result/mask_scheduling/test_{idx}.png", nrow=10)
            vutils.save_image(imga, f"./Result/imga/test_{idx}.png", nrow=7)


class MaskedImage:
    def __init__(self, args):
        mi_ori=LoadTestData(root= args.test_maskedimage_path, partial=args.partial)
        self.mi_ori =  DataLoader(mi_ori,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            drop_last=True,
                            pin_memory=True,
                            shuffle=False)
        mask_ori =LoadMaskData(root= args.test_mask_path, partial=args.partial)
        self.mask_ori =  DataLoader(mask_ori,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            drop_last=True,
                            pin_memory=True,
                            shuffle=False)
        self.device=args.device

    def get_mask_latent(self, mask):
        mask = torch.nn.functional.avg_pool2d(mask, kernel_size=4, stride=4)  # 64x64 → 16x16
        mask = (mask[0][0] >= 1.0).float()  # 保留1.0為有效區域
        mask_b = (mask == 0).bool().flatten().unsqueeze(0).to(self.device)
        return mask_b


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT for Inpainting")
    parser.add_argument('--device'     , type=str, default="cuda:0", help='Which device the training is on.')# default="cuda:0"
    parser.add_argument('--batch-size' , type=int, default=1, help='Batch size for testing.')
    parser.add_argument('--partial'    , type=float, default=1.0, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker')
    
    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for MaskGIT')
    
    
#TODO3 step1-2: modify the path, MVTM parameters
    parser.add_argument('--load-transformer-ckpt-path', type=str, default='transformer_checkpoints/ckpt_100.pt', help='load ckpt')
    
    #dataset path
    parser.add_argument('--test-maskedimage-path', type=str, default='./lab3_dataset/masked_image', help='Path to testing image dataset.')
    parser.add_argument('--test-mask-path'       , type=str, default='./lab3_dataset/mask64', help='Path to testing mask dataset.')
    #MVTM parameter
    parser.add_argument('--sweet-spot', type=int, default=4       , help='sweet spot: the best step in total iteration')
    parser.add_argument('--total-iter', type=int, default=4       , help='total step for mask scheduling')
    parser.add_argument('--mask-func' , type=str, default='cosine', help='mask scheduling function')

    args = parser.parse_args()

    t=MaskedImage(args)
    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    MaskGit_CONFIGS["model_param"]['gamma_type'] = args.mask_func
    maskgit = MaskGIT(args, MaskGit_CONFIGS)

    for i, (image, mask) in enumerate(tqdm(zip(t.mi_ori, t.mask_ori), total=len(t.mi_ori), desc="Inpainting")):
        image = image.to(args.device)
        mask = mask.to(args.device)
        mask_b = t.get_mask_latent(mask)
        maskgit.inpainting(image, mask_b, i)