"""
Author: Chi-An Chen
Date: 2025-07-17
Description: Training
"""

import os
import argparse
import yaml
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import MaskGit as VQGANTransformer
from utils import LoadTrainData


class TrainTransformer:
    def __init__(self, args, config):
        self.args = args
        self.device = args.device
        self.model = VQGANTransformer(config["model_param"]).to(self.device)
        self.optimizer, self.scheduler = self.configure_optimizers()
        os.makedirs("transformer_checkpoints", exist_ok=True)

    def check_model(self):
        from thop import profile
        x = torch.randn(1, 3, 64, 64).to(self.device)
        ratio = np.random.rand()
        flops, params = profile(self.model, inputs=(x, ratio))
        print("=" * 50)
        print(f"FLOPs: {flops / 1e9:.4f} GFLOPs")
        print(f"Params: {params / 1e6:.4f} M")
        print("=" * 50)

    def train_one_epoch(self, loader, epoch):
        self.model.train()
        losses = []
        pbar = tqdm(loader, desc=f"(Train) Epoch {epoch}", ncols=120)

        for x in pbar:
            x = x.to(self.device)
            self.optimizer.zero_grad()

            ratio = np.random.rand()
            y_pred, y = self.model(x, ratio)
            loss = F.cross_entropy(y_pred, y)
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())
            pbar.set_postfix(loss=loss.item(), lr=self.scheduler.get_last_lr()[0])

        self.scheduler.step()
        return np.mean(losses)

    def eval_one_epoch(self, loader, epoch):
        self.model.eval()
        losses = []
        pbar = tqdm(loader, desc=f"(Val) Epoch {epoch}", ncols=120)

        with torch.no_grad():
            for x in pbar:
                x = x.to(self.device)
                ratio = np.random.rand()
                y_pred, y = self.model(x, ratio)
                loss = F.cross_entropy(y_pred, y)
                losses.append(loss.item())
                pbar.set_postfix(loss=loss.item(), lr=self.scheduler.get_last_lr()[0])

        return np.mean(losses)

    def configure_optimizers(self):
        decay, no_decay = set(), set()
        whitelist = (torch.nn.Linear,)
        blacklist = (torch.nn.LayerNorm, torch.nn.Embedding)

        for mn, m in self.model.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist):
                    no_decay.add(fpn)

        no_decay.add("pos_emb")

        param_dict = {pn: p for pn, p in self.model.transformer.named_parameters()}
        assert decay.isdisjoint(no_decay), "Parameters in both decay/no_decay!"
        assert param_dict.keys() == decay.union(no_decay), "Some parameters were not categorized!"

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.args.learning_rate, betas=(0.9, 0.95))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: min((step + 1) / self.args.warmup_steps, 1)
        )

        return optimizer, scheduler


def main():
    parser = argparse.ArgumentParser(description="Train MaskGIT Transformer")
    parser.add_argument('--train_d_path', type=str, default="./cat_face/train/")
    parser.add_argument('--val_d_path', type=str, default="./cat_face/val/")
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt')
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--partial', type=float, default=1.0)
    parser.add_argument('--accum-grad', type=int, default=10)

    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--save-per-epoch', type=int, default=5)
    parser.add_argument('--start-from-epoch', type=int, default=0)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--warmup-steps', type=int, default=1)
    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml')

    args = parser.parse_args()
    config = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    trainer = TrainTransformer(args, config)

    train_loader = DataLoader(
        LoadTrainData(args.train_d_path, partial=args.partial),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        shuffle=True
    )
    val_loader = DataLoader(
        LoadTrainData(args.val_d_path, partial=args.partial),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        shuffle=False
    )

    trainer.check_model()

    for epoch in range(args.start_from_epoch + 1, args.epochs + 1):
        train_loss = trainer.train_one_epoch(train_loader, epoch)
        val_loss = trainer.eval_one_epoch(val_loader, epoch)

        print(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | LR: {trainer.scheduler.get_last_lr()[0]:.6f}")

        if epoch % args.save_per_epoch == 0:
            save_path = f"transformer_checkpoints/ckpt_{epoch}.pt"
            torch.save(trainer.model.transformer.state_dict(), save_path)
            print(f"Model saved at: {save_path}")


if __name__ == '__main__':
    main()