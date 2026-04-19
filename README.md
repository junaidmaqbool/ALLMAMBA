# 🐍 Mamba Models — Retinal Eye Image HB Estimation

All Mamba variants adapted for **dual-task learning**:
- ✅ **Classification** — Anemic (HB < 12.0) vs Non-Anemic
- ✅ **Regression** — Predict exact HB value in g/dL

---

## Folder Structure

```
MAMBA_MODELS/
├── 01_Mamba_Official/          ← state-spaces/mamba (Mamba1 + Mamba2 + Mamba3)
│   └── mamba_eye_hb_notebook.ipynb
│
├── 02_VMamba/                  ← VMamba (2D Selective Scan, ICLR 2025)
│   └── vmamba_eye_hb_notebook.ipynb
│
├── 03_MambaVision/             ← MambaVision NVIDIA (CVPR 2025, hybrid Mamba+Transformer)
│   └── mambavision_eye_hb_notebook.ipynb
│
├── 04_MedMamba/                ← MedMamba (medical image classification)
│   └── medmamba_eye_hb_notebook.ipynb
│
├── 05_VSSD_Mamba2Vision/       ← VSSD (Mamba2 for vision, ICCV 2025)
│   └── [use mamba_eye_hb_notebook with mamba_ver="mamba2"]
│
├── 06_Mamba3_Minimal/          ← Mamba3 minimal PyTorch implementation
│   └── [use mamba_eye_hb_notebook with mamba_ver="mamba3"]
│
└── 07_DSA_Mamba_Custom/        ← DSA-Mamba (custom for HB estimation)
    ├── dsa_mamba.py
    └── dsa_mamba_eye_hb_notebook.ipynb
```

---

## Quick Start on Kaggle

```python
# 1. Clone this repo
!git clone https://github.com/YOUR_USERNAME/MAMBA_HB_MODELS.git

# 2. Install deps
!pip install mamba-ssm causal-conv1d einops timm mambavision pandas openpyxl scikit-learn tqdm

# 3. Open any notebook and set paths:
CFG = dict(
    image_dir = "/kaggle/input/datasets/junaidgpu/imagehb/dataset/dataset/left_eye",
    csv_path  = "/kaggle/input/datasets/junaidgpu/imagehb/dataset/dataset/merge_excel_1.xlsx",
    image_col = "Patient ID",
    hb_col    = "HB",
    hb_thresh = 12.0,
    ...
)
```

---

## Model Comparison

| Model | Architecture | Params | Speed | Notes |
|---|---|---|---|---|
| Mamba1 | SSM | ~20M | Fast | Baseline SSM |
| Mamba2 | SSD | ~20M | Faster | ICML 2024 |
| Mamba3 | MIMO+Rotary | ~22M | Fast | ICLR 2026 |
| VMamba | 2D cross-scan SSM | ~31M | Medium | Best for 2D vision |
| MambaVision | Mamba+Transformer | ~31M | Medium | CVPR 2025, pretrained |
| MedMamba | SS-Conv-SSM | ~28M | Medium | Medical domain |
| DSA-Mamba | Spectral Attn+SSM | ~18M | Fast | Custom for HB |

---

## Loss Function

```python
total_loss = cls_weight * CrossEntropy(logits, labels)
           + reg_weight * MSE(hb_pred, hb_true)
```

Tune `cls_weight` and `reg_weight` in `CFG` per model.

---

## Data Format

- **Images:** Eye images named by Patient ID (e.g. `001.jpg`)
- **CSV/Excel:** Must have `Patient ID` and `HB` columns
- **HB threshold:** Default 12.0 g/dL (anemic below, normal above)

---

## GitHub Push

```bash
cd MAMBA_MODELS
git init
git add .
git commit -m "All Mamba models for HB estimation"
git remote add origin https://github.com/YOUR_USERNAME/mamba-hb-models.git
git push -u origin main
```
