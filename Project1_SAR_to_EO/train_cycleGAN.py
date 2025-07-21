#!/usr/bin/env python
# coding: utf-8

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# ---------------------------
# Config
# ---------------------------
class Config:
    batch_size = 4
    num_epochs = 5
    lr = 2e-4
    beta1 = 0.5
    beta2 = 0.999
    image_size = 256
    checkpoint_dir = './checkpoints'
    output_dir = './outputs'
    data_dir = '/kaggle/input/6000sarv2/preprocessed'  # ✅ Update if needed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = Config()
os.makedirs(cfg.checkpoint_dir, exist_ok=True)
os.makedirs(cfg.output_dir, exist_ok=True)

print(f"✅ Using device: {cfg.device}")

# ---------------------------
# Models
# ---------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, in_channels=2, out_channels=3, n_res_blocks=6):
        super().__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, kernel_size=7),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        ]
        curr_dim = 64
        for _ in range(2):
            model += [
                nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1),
                nn.InstanceNorm2d(curr_dim * 2),
                nn.ReLU(True)
            ]
            curr_dim *= 2

        for _ in range(n_res_blocks):
            model += [ResidualBlock(curr_dim)]

        for _ in range(2):
            model += [
                nn.ConvTranspose2d(curr_dim, curr_dim // 2, 4, 2, 1),
                nn.InstanceNorm2d(curr_dim // 2),
                nn.ReLU(True)
            ]
            curr_dim = curr_dim // 2

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(curr_dim, out_channels, 7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        def block(in_c, out_c, norm=True):
            layers = [nn.Conv2d(in_c, out_c, 4, 2, 1)]
            if norm:
                layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, 64, norm=False),
            *block(64, 128),
            *block(128, 256),
            nn.Conv2d(256, 1, kernel_size=4, padding=1)
        )

    def forward(self, x):
        return self.model(x)

# ---------------------------
# Dataset
# ---------------------------
class PreprocessedDataset(Dataset):
    def __init__(self, root):
        self.paths = sorted([os.path.join(root, f) for f in os.listdir(root) if f.endswith('.pt')])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return torch.load(self.paths[idx])

# ---------------------------
# Utils
# ---------------------------
def init_weights(net):
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def save_sample(real_A, fake_B, real_B, path):
    real_A_rgb = real_A[:, :1].repeat(1, 3, 1, 1)
    grid = torch.cat([real_A_rgb, fake_B, real_B], dim=3)
    vutils.save_image(grid, path, normalize=True)

# ---------------------------
# Train
# ---------------------------
def train():
    dataset = PreprocessedDataset(cfg.data_dir)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
    print(f"✅ Loaded {len(dataset)} samples.")

    G = Generator().to(cfg.device)
    D = Discriminator().to(cfg.device)
    init_weights(G)
    init_weights(D)

    criterion_GAN = nn.MSELoss()
    criterion_L1 = nn.L1Loss()

    optimizer_G = torch.optim.Adam(G.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))

    for epoch in range(cfg.num_epochs):
        for i, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}")):
            real_A = batch['A'].to(cfg.device)
            real_B = batch['B'].to(cfg.device)

            # Train D
            optimizer_D.zero_grad()
            fake_B = G(real_A).detach()
            loss_D_real = criterion_GAN(D(real_B), torch.ones_like(D(real_B)))
            loss_D_fake = criterion_GAN(D(fake_B), torch.zeros_like(D(fake_B)))
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()

            # Train G
            optimizer_G.zero_grad()
            fake_B = G(real_A)
            loss_G_GAN = criterion_GAN(D(fake_B), torch.ones_like(D(fake_B)))
            loss_G_L1 = criterion_L1(fake_B, real_B) * 100.0
            loss_G = loss_G_GAN + loss_G_L1
            loss_G.backward()
            optimizer_G.step()

            if i % 100 == 0:
                save_sample(real_A[:1], fake_B[:1], real_B[:1], f"{cfg.output_dir}/sample_e{epoch}_i{i}.png")

        torch.save(G.state_dict(), f"{cfg.checkpoint_dir}/G_epoch{epoch}.pt")
        torch.save(D.state_dict(), f"{cfg.checkpoint_dir}/D_epoch{epoch}.pt")

    print("✅ Training finished!")

if __name__ == "__main__":
    train()
