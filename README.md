# SAR → EO Image Translation Using CycleGAN 

## ✏️ **Project Overview**
This project translates Sentinel-1 SAR satellite images into Sentinel-2 optical (EO) images using an enhanced CycleGAN framework.  
The goal: generate realistic EO data even when optical images are missing or cloud-covered — useful for remote sensing, environmental monitoring, and data augmentation.

---

## ⚙️ **Architecture & Methods**
### ✅ Coarse-to-fine generator
- Captures **global structure** first, then refines with **residual blocks** to add fine details.
- Helps produce sharper, more realistic EO images.

### ✅ Multi-scale discriminators
- Three discriminators see the generated EO at:
  - Full resolution (256×256)
  - Half resolution (128×128)
  - Quarter resolution (64×64)
- Each discriminator checks realism at its own scale:
  - Large scale: color & layout
  - Small scale: texture & fine details

### ✅ Perceptual loss
- Uses pretrained VGG16 features.
- Makes generated images perceptually closer to real EO, beyond pixel-level similarity.

### ✅ Feature matching loss
- Forces generator to match **intermediate feature activations** of discriminators for real vs fake.
- Helps stabilize GAN training and improves structure.

### ✅ Denoising pre-filter (Non-Local Means)
- SAR images contain **speckle noise** (random granular patterns from radar interference).
- Applying NLM filter before generator removes noise while preserving edges.
- Leads to higher PSNR and SSIM.

---

## 📦 **Dataset**
- ~6000 `.pt` files, each containing:
  - `'A'`: SAR patch (2 channels: VV, VH), shape (2×256×256)
  - `'B'`: EO patch (3 channels: RGB), shape (3×256×256)
- Normalized to [-1,1].

---

## 🧪 **Evaluation metrics**
- **PSNR** (Peak Signal-to-Noise Ratio): measures pixel-level fidelity.
- **SSIM** (Structural Similarity): measures perceptual and structural similarity.
- **NDVI** (Normalized Difference Vegetation Index): checks if generated EO preserves vegetation info.

Example (after training):
| Sample | SSIM  | PSNR (dB) | NDVI  |
|------:|:-----:|:---------:|:-----:|
| avg   | 0.51  | 22.95     | 0.20  |

*After full training, expected PSNR ~25–28 dB, SSIM ~0.6–0.75.*

---

## 🏗 **How it works (step by step)**
1. Load noisy SAR images.
2. Apply **Non-Local Means denoising**.
3. Pass through **coarse-to-fine generator** → generate EO images.
4. Three **multi-scale discriminators** judge realism.
5. Generator optimized using:
   - GAN loss
   - Perceptual loss
   - Feature matching loss
   - L1 loss

---

