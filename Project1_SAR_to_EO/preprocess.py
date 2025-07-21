import os
from glob import glob
import rasterio
import numpy as np
import torch
from tqdm import tqdm

# ✏️ CONFIG: put your directories here
sar_dir  = r"D:\Kinshouks_data\ROIs2017_winter_s1\ROIs2017_winter"
eo_dir   = r"D:\Kinshouks_data\ROIs2017_winter_s2\ROIs2017_winter"
max_samples = 6000

# 🔍 Find files
sar_paths = sorted(glob(os.path.join(sar_dir, '*.tif')))
eo_paths  = sorted(glob(os.path.join(eo_dir, '*.tif')))

if not sar_paths or not eo_paths:
    print("❌ No files found in one or both dirs.")
    exit()

n = min(len(sar_paths), len(eo_paths), max_samples)
print(f"✅ Found {len(sar_paths)} SAR & {len(eo_paths)} EO images. Will preprocess {n} pairs.")

# ⚙️ Normalization
def normalize(img):
    img = img.astype(np.float32)
    return 2 * ((img - np.min(img)) / (np.ptp(img) + 1e-6)) - 1

# 📦 Store preprocessed data here
data = []

# 🔁 Preprocess on the fly with progress
for i in tqdm(range(n), desc="Preprocessing"):
    try:
        with rasterio.open(sar_paths[i]) as sar_ds:
            sar = sar_ds.read([1, 2])  # shape: (2, H, W)
            sar = normalize(sar)
            sar_tensor = torch.from_numpy(sar)

        with rasterio.open(eo_paths[i]) as eo_ds:
            eo = eo_ds.read([4, 3, 2])  # shape: (3, H, W)
            eo = normalize(eo)
            eo_tensor = torch.from_numpy(eo)

        # Append as tuple
        data.append( (sar_tensor, eo_tensor) )

    except Exception as e:
        print(f"\n❌ Error at index {i}: {e}\nSAR: {sar_paths[i]}\nEO : {eo_paths[i]}")

print(f"✅ Done! Preprocessed {len(data)} samples.")


# import os
# import torch
# from glob import glob
# import rasterio
# import numpy as np
# from tqdm import tqdm

# # === 📂 Path settings ===
# # Change these according to your local repo structure
# repo_root = r"D:\MyLocalRepo"       # your local cloned repo
# sar_dir   = os.path.join(repo_root, "data", "ROIs2017_winter_s1", "ROIs2017_winter")
# eo_dir    = os.path.join(repo_root, "data", "ROIs2017_winter_s2", "ROIs2017_winter")
# save_dir  = os.path.join(repo_root, "preprocessed")

# os.makedirs(save_dir, exist_ok=True)

# # === 🔍 Find files ===
# sar_paths = sorted(glob(os.path.join(sar_dir, '*.tif')))
# eo_paths  = sorted(glob(os.path.join(eo_dir, '*.tif')))

# print(f"✅ Found {len(sar_paths)} SAR images")
# print(f"✅ Found {len(eo_paths)} EO  images")

# if len(sar_paths) == 0 or len(eo_paths) == 0:
#     print("❌ No input files found in local repo.")
#     exit()

# max_samples = min(len(sar_paths), len(eo_paths), 6000)
# print(f"🔁 Preprocessing {max_samples} SAR-EO image pairs...")

# # === ⚙️ Preprocessing function ===
# def normalize_image(img):
#     img = img.astype(np.float32)
#     img_min = np.min(img)
#     img_max = np.max(img)
#     img = (img - img_min) / (img_max - img_min + 1e-6)  # [0,1]
#     img = 2 * img - 1  # [-1,1]
#     return img

# # === 💾 Process and save ===
# for i in tqdm(range(max_samples), desc="Saving preprocessed samples"):
#     sar_path = sar_paths[i]
#     eo_path  = eo_paths[i]

#     try:
#         with rasterio.open(sar_path) as sar_img:
#             sar = sar_img.read([1, 2])  # Shape: (2, H, W)
#             sar = normalize_image(sar)
#             sar_tensor = torch.from_numpy(sar)

#         with rasterio.open(eo_path) as eo_img:
#             eo = eo_img.read([4, 3, 2])  # Shape: (3, H, W)
#             eo = normalize_image(eo)
#             eo_tensor = torch.from_numpy(eo)

#         sample = {'A': sar_tensor, 'B': eo_tensor}
#         torch.save(sample, os.path.join(save_dir, f'sample_{i:04d}.pt'))

#     except Exception as e:
#         print(f"\n❌ Error at index {i}:\nSAR: {sar_path}\nEO : {eo_path}\n{e}")

# print("✅ Preprocessing complete. Files saved to:", save_dir)
