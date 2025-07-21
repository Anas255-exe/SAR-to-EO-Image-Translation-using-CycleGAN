

# Project 1: SAR → EO Image Translation Using CycleGAN

## Team Members  
- **Gurmehar** (23/CS/162)  
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
   
<img width="1755" height="739" alt="image" src="https://github.com/user-attachments/assets/d84c704e-7044-426b-94b8-1d3457510c3d" />




  

2. **Figure 2. Additional Examples**  
3. **Figure 3. Quantitative Metrics Over Samples**  

### 📊 Evaluation Metrics (after 30 epochs)

| Sample             | SSIM   | PSNR   | NDVI  |
|-------------------:|:------:|:------:|:-----:|
| sample_0000.pt     | 0.7110 | 24.58  | 0.5067 |
| sample_0001.pt     | 0.6143 | 17.03  | 0.4683 |
| sample_0002.pt     | 0.7328 | 26.68  | 0.4780 |
| sample_0003.pt     | 0.6836 | 21.49  | 0.5638 |
| sample_0004.pt     | 0.6749 | 26.36  | 0.5649 |
| sample_0005.pt     | 0.7644 | 25.68  | 0.5153 |
| sample_0006.pt     | 0.7750 | 26.93  | 0.5481 |
| sample_0007.pt     | 0.6369 | 19.69  | 0.5864 |
| sample_0008.pt     | 0.6452 | 19.65  | 0.6178 |
| sample_0009.pt     | 0.7290 | 21.83  | 0.6507 |
| sample_0010.pt     | 0.7109 | 22.55  | 0.6260 |

✅ **Overall Performance:**
- **Average SSIM:** 0.6980
- **Average PSNR:** 22.95 dB
- **Average NDVI:** 0.5569


4. **Figure 4. Training Curves **  
  <img width="1108" height="834" alt="image" src="https://github.com/user-attachments/assets/8b11b8a6-cc6d-480b-950b-daad6bf699c3" />
  <img width="1147" height="908" alt="image" src="https://github.com/user-attachments/assets/da1a4308-2736-40ae-81c3-e477750e7ee2" />



---

## 7. Key Findings & Observations  
- Normalizing each channel independently to [–1,1] stabilized training.  
- Achieved average SSIM ≈ 0.69 and PSNR ≈ 22.95 dB on RGB.  
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


---
