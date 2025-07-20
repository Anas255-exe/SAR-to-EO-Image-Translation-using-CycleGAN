

# Project 1: SAR → EO Image Translation Using CycleGAN

## Team Members  
- **Gurmehar** (23/CS/168)  
- **Mohammad Anas**  (23/CS/255)  

---

## 1. Project Overview  
We implement a CycleGAN to translate Sentinel-1 SAR imagery (2 channels) into Sentinel-2 EO imagery (3 channels).  
  
Objectives:  
- Normalize and preprocess SAR & EO patches  
- Train CycleGAN under the SAR→EO (RGB) configuration  
- Evaluate results with spectral SSIM, PSNR, and visual comparisons  

---

## 2. Repository Structure  
```
project/
├── Project1_SAR_to_EO/
│   ├── data/
│   │   ├── raw/            # original .pt samples  
│   │   └── processed/      # normalized train/val splits  
│   ├── checkpoints/        # saved model weights  
│   ├── generated_samples/  # output EO images & comparisons  
│   ├── preprocess.py       # preprocessing & splitting  
│   ├── train_cycleGAN.py   # training script  
│   ├── evaluate_results.py # metrics & visualization  
│   └── config.yaml         # hyperparameters & paths  
├── README.md               
└── requirements.txt        
```

---

## 3. Setup & Installation  
1. Clone the repo:  
   ```
   git clone https://github.com/your‐org/your‐repo.git
   cd your‐repo/Project1_SAR_to_EO
   ```  
2. Create a virtual environment and install dependencies:  
   ```
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```  

---

## 4. Data Preprocessing  
1. Place your 6,000 `.pt` files in `data/raw/`. Each file must be a dict:  
   ```python
   {'A': Tensor(2×256×256),   # SAR (VV, VH)
    'B': Tensor(3×256×256)}   # EO (RGB)
   ```  
2. Run preprocessing and train/val split (80/20):  
   ```
   python preprocess.py
   ```  
   - Normalizes each channel to [–1,1]  
   - Saves to `data/processed/train/` and `data/processed/val/`  

---

## 5. Training  
Edit `config.yaml` to adjust hyperparameters (batch size, epochs, learning rate). Then run:  
```
python train_cycleGAN.py
```  
— Model checkpoints will be saved under `checkpoints/` as `G_A2B_epoch{n}.pth`.  

---

## 6. Evaluation & Visualizations  
Run metrics computation and generate example images:  
```
python evaluate_results.py
```  
This will:  
- Compute SSIM & PSNR for N validation samples  
- Save comparison grids to `generated_samples/`  

### 6.1 Suggested Visuals  
Leave space below to paste or link your visuals:

1. **Figure 1. SAR → EO Example (Before / After)**  
   _Description:_ Show one SAR input band (VV or composite), the real EO RGB, and the generated EO RGB side-by-side.  



  <img width="2076" height="745" alt="image" src="https://github.com/user-attachments/assets/f3b62ba1-9ee5-4393-a8eb-276d6ed7fd2a" />

2. **Figure 2. Additional Examples**  
3. **Figure 3. Quantitative Metrics Over Samples**  
   _Description:_ A table or bar chart showing SSIM and PSNR for each selected sample.  
   | Sample ID | SSIM  | PSNR  |
   |-----------|-------|-------|
   | sample_001| 0.76  | 19.2 dB |
   | sample_042| 0.81  | 20.5 dB |
   | …         | …     | …      |

4. **Figure 4. Training Curves **  
   _Description:_ Plot of generator and discriminator losses vs. epochs, as well as SSIM/PSNR on the validation set over time.  
   ![Figure 4: placeholder](checkpoints/training_curves.png)  

---

## 7. Key Findings & Observations  
- Normalizing each channel independently to [–1,1] stabilized training.  
- Achieved average SSIM ≈ 0.78 and PSNR ≈ 20 dB on RGB.  
- Failure modes: difficulty reconstructing fine texture in heavily vegetated zones.  

---

## 8. Tools and Frameworks  
- Python 3.8+  
- PyTorch  
- torchvision  
- scikit-image (SSIM, PSNR)  
- matplotlib  

---

## 9. Next Steps (Future Work)  
- Experiment with perceptual (VGG) loss for sharper textures  
- Extend to NIR or SWIR target bands  
- Integrate cloud-mask guided loss for cloudy regions  

---

## 10. How to Reproduce  
1. Ensure data in `data/raw/`  
2. Install requirements and follow Sections 4–6 above  
3. Inspect `config.yaml` for all path and hyperparameter settings  

---
