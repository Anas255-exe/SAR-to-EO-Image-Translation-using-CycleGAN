# evaluate_results.py
import os
import torch
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import numpy as np

# ---------------- Config ----------------
class Config:
    checkpoint_path = '/workspaces/SAR-to-EO-Image-Translation-using-CycleGAN/Project1_SAR_to_EO/checkpoints/G_epoch29.pt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_data_dir = '/workspaces/SAR-to-EO-Image-Translation-using-CycleGAN/Project1_SAR_to_EO/Data/Preprocessed'   # folder containing sample .pt files with {'A': SAR, 'B': EO}
    output_dir = 'eval_outputs_last'

cfg = Config()

os.makedirs(cfg.output_dir, exist_ok=True)

# ---------------- Load model ----------------
from train_cycleGAN import Generator   # adjust import if your Generator is defined elsewhere

G = Generator().to(cfg.device)
G.load_state_dict(torch.load(cfg.checkpoint_path, map_location=cfg.device))
G.eval()

print(f"Loaded Generator from: {cfg.checkpoint_path}")

# ---------------- Load test sample.pt files ----------------
sample_files = sorted([f for f in os.listdir(cfg.test_data_dir) if f.endswith('.pt')])
print(f"Found {len(sample_files)} test samples.")

# Metrics lists
ssim_list = []
psnr_list = []
ndvi_list = []

for idx, fname in enumerate(sample_files):
    data = torch.load(os.path.join(cfg.test_data_dir, fname))
    sar_tensor = data['A'].to(cfg.device)  # SAR input
    gt_tensor = data['B'].cpu()            # ground truth EO

    if sar_tensor.dim() == 3:
        sar_tensor = sar_tensor.unsqueeze(0)

    with torch.no_grad():
        fake_eo = G(sar_tensor)[0].cpu()  # remove batch dim

    # Denormalize [-1,1] → [0,1]
    fake_eo_img = (fake_eo + 1) / 2
    gt_img = (gt_tensor + 1) / 2

    # Convert to numpy
    fake_eo_np = fake_eo_img.permute(1,2,0).numpy()
    gt_np = gt_img.permute(1,2,0).numpy()

    # ---------------- Metrics ----------------
    # Spectral-wise SSIM (mean over channels)
    ssim_channels = [ssim(gt_np[:,:,c], fake_eo_np[:,:,c], data_range=1) for c in range(fake_eo_np.shape[2])]
    mean_ssim = np.mean(ssim_channels)
    ssim_list.append(mean_ssim)

    # PSNR
    mse = np.mean((gt_np - fake_eo_np) ** 2)
    psnr_value = 20 * np.log10(1.0 / np.sqrt(mse + 1e-8))
    psnr_list.append(psnr_value)

    # NDVI: assume last channel=NIR, first channel=Red
    if fake_eo_np.shape[2] >= 2:
        nir = fake_eo_np[:,:,-1]
        red = fake_eo_np[:,:,0]
        ndvi = (nir - red) / (nir + red + 1e-8)
        mean_ndvi = np.mean(ndvi)
        ndvi_list.append(mean_ndvi)

    # ---------------- Save visualization ----------------
    fig, axs = plt.subplots(1,2, figsize=(8,4))
    axs[0].imshow(gt_np)
    axs[0].set_title('Ground Truth EO')
    axs[0].axis('off')

    axs[1].imshow(fake_eo_np)
    axs[1].set_title('Generated EO')
    axs[1].axis('off')

    plt.tight_layout()
    save_path = os.path.join(cfg.output_dir, f'{fname[:-3]}.png')
    plt.savefig(save_path)
    plt.close(fig)

    print(f"Processed {fname} | SSIM: {mean_ssim:.4f} | PSNR: {psnr_value:.2f} | NDVI: {mean_ndvi:.4f}")

# ---------------- Print overall metrics ----------------
print("\n=== Overall Performance ===")
print(f"Average SSIM: {np.mean(ssim_list):.4f}")
print(f"Average PSNR: {np.mean(psnr_list):.2f}")
print(f"Average NDVI: {np.mean(ndvi_list):.4f}")
