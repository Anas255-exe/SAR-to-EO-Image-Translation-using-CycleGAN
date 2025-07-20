import os
import torch
from glob import glob
import rasterio
import numpy as np
from tqdm import tqdm

sar_dir = r"D:\Kinshouks_data\ROIs2017_winter_s1\ROIs2017_winter"
eo_dir  = r"D:\Kinshouks_data\ROIs2017_winter_s2\ROIs2017_winter"
save_dir = r"D:\Kinshouks_data\preprocessed"
os.makedirs(save_dir, exist_ok=True)

sar_paths = sorted(glob(os.path.join(sar_dir, '', '*.tif'), recursive=True))
eo_paths  = sorted(glob(os.path.join(eo_dir, '', '*.tif'), recursive=True))

print(f"✅ Found {len(sar_paths)} SAR images")
print(f"✅ Found {len(eo_paths)} EO  images")

if len(sar_paths) == 0 or len(eo_paths) == 0:
    print("❌ No input files found.")
    exit()

max_samples = min(len(sar_paths), len(eo_paths), 6000)
print(f"🔁 Preprocessing {max_samples} SAR-EO image pairs by index...")

def normalize_image(img):
    img = img.astype(np.float32)
    img_min = np.min(img)
    img_max = np.max(img)
    img = (img - img_min) / (img_max - img_min + 1e-6)  # [0,1]
    img = 2 * img - 1  # [-1,1]
    return img

for i in tqdm(range(max_samples), desc="Saving preprocessed samples"):
    sar_path = sar_paths[i]
    eo_path  = eo_paths[i]

    try:
        with rasterio.open(sar_path) as sar_img:
            sar = sar_img.read([1, 2])  # Shape: (2, H, W)
            sar = normalize_image(sar)
            sar_tensor = torch.from_numpy(sar)

        with rasterio.open(eo_path) as eo_img:
            eo = eo_img.read([4, 3, 2])  # Shape: (3, H, W)
            eo = normalize_image(eo)
            eo_tensor = torch.from_numpy(eo)

        sample = {'A': sar_tensor, 'B': eo_tensor}
        torch.save(sample, os.path.join(save_dir, f'sample_{i:04d}.pt'))

    except Exception as e:
        print(f"\n❌ Error at index {i}:\nSAR: {sar_path}\nEO : {eo_path}\n{e}")

print("✅ Preprocessing complete.")
