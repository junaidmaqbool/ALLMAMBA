"""
train_hb.py — DSA-Mamba adapted for Eye Image HB Estimation
=============================================================
Extends the original DSA-Mamba (First-Ronin/DSA-Mamba) with:
  ① Binary classification : Anemic (HB < threshold) vs Non-Anemic
  ② Regression            : Predict exact HB value (g/dL)

Usage (Kaggle):
    python train_hb.py \
        --train-dataset-path /kaggle/input/.../left_eye \
        --csv-path /kaggle/input/.../merge_excel_1.xlsx \
        --image-col "Patient ID" \
        --hb-col "HB" \
        --hb-threshold 12.0 \
        --batch-size 4 \
        --epochs 50
"""

import os, sys, json, argparse, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, roc_auc_score,
                              mean_absolute_error, mean_squared_error)
from tqdm import tqdm

# ── Import official DSA-Mamba backbone ────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from model.DSAmamba import VSSM as DSAMambaBackbone


# ──────────────────────────────────────────────────────────────────────────────
# 1. Dual-head model wrapper
# ──────────────────────────────────────────────────────────────────────────────

class DSAMambaDualHead(nn.Module):
    """
    Official DSA-Mamba backbone + dual output heads.

    backbone.head is replaced with:
      - cls_head : classification  (B → num_classes)
      - reg_head : regression      (B → 1)
    """
    def __init__(self, in_chans: int = 3, num_classes: int = 2,
                 in_depths=(2,2,4), out_depths=(2,2),
                 in_dims=(96,192,384), out_dims=(768,384)):
        super().__init__()
        self.backbone = DSAMambaBackbone(
            in_chans=in_chans,
            num_classes=0,        # disable original head
            in_depths=list(in_depths),
            out_depths=list(out_depths),
            in_dims=list(in_dims),
            out_dims=list(out_dims),
        )
        feat_dim = list(out_dims)[-1]   # last decoder dim

        self.cls_head = nn.Linear(feat_dim, num_classes)
        self.reg_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feat_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor):
        """
        Returns:
            logits  (B, num_classes)
            hb_pred (B, 1)
        """
        feat = self.backbone.forward_backbone(x)          # (B, H', W', C)
        feat = feat.permute(0, 3, 1, 2)                   # (B, C, H', W')
        feat = self.backbone.avgpool(feat)                 # (B, C, 1, 1)
        feat = torch.flatten(feat, start_dim=1)           # (B, C)

        return self.cls_head(feat), self.reg_head(feat)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Dataset
# ──────────────────────────────────────────────────────────────────────────────

class EyeHBDataset(Dataset):
    def __init__(self, df, image_dir, image_col, hb_col, hb_thresh, transform=None):
        self.df        = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.image_col = image_col
        self.hb_col    = hb_col
        self.hb_thresh = hb_thresh
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        pid   = str(row[self.image_col])
        hb    = float(row[self.hb_col])
        label = 0 if hb < self.hb_thresh else 1

        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ""]:
            p = os.path.join(self.image_dir, pid + ext)
            if os.path.exists(p):
                img_path = p
                break
        if img_path is None:
            raise FileNotFoundError(f"No image found for Patient ID '{pid}' in {self.image_dir}")

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long), torch.tensor([[hb]], dtype=torch.float32)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Metrics
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(model, loader, device, ce_fn, mse_fn, cls_w, reg_w):
    model.eval()
    val_loss = correct = total = 0
    all_preds, all_labels, all_scores = [], [], []
    all_hb_pred, all_hb_true = [], []

    with torch.no_grad():
        for imgs, labels, hb_true in loader:
            imgs, labels, hb_true = imgs.to(device), labels.to(device), hb_true.to(device)
            logits, hb_pred = model(imgs)
            loss = cls_w * ce_fn(logits, labels) + reg_w * mse_fn(hb_pred, hb_true)
            val_loss += loss.item()

            probs  = torch.softmax(logits, dim=1)
            preds  = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(probs[:, 1].cpu().numpy())
            all_hb_pred.extend(hb_pred.cpu().squeeze().tolist())
            all_hb_true.extend(hb_true.cpu().squeeze().tolist())

    val_loss /= len(loader)
    acc  = correct / total
    try:
        auc = roc_auc_score(all_labels, all_scores)
    except Exception:
        auc = float("nan")
    mae  = mean_absolute_error(all_hb_true, all_hb_pred)
    rmse = math.sqrt(mean_squared_error(all_hb_true, all_hb_pred))

    return val_loss, acc, auc, mae, rmse, all_preds, all_labels, all_hb_pred, all_hb_true


# ──────────────────────────────────────────────────────────────────────────────
# 4. Args
# ──────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser("DSA-Mamba HB Estimation")
    p.add_argument("--train-dataset-path", required=True, help="Folder with eye images")
    p.add_argument("--csv-path",           required=True, help="Excel/CSV with Patient ID + HB")
    p.add_argument("--image-col",  default="Patient ID", help="Column name for image filename (no ext)")
    p.add_argument("--hb-col",     default="HB",         help="Column name for HB value")
    p.add_argument("--hb-threshold", type=float, default=12.0, help="Anemia threshold g/dL")
    p.add_argument("--val-split",    type=float, default=0.2)
    p.add_argument("--batch-size",   type=int,   default=4)
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--num-works",    type=int,   default=2)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--cls-weight",   type=float, default=1.0, help="Classification loss weight")
    p.add_argument("--reg-weight",   type=float, default=0.5, help="Regression loss weight")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--save-path",    default="./pth_out/dsamamba_hb_best.pth")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# 5. Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────────
    if args.csv_path.endswith(".xlsx") or args.csv_path.endswith(".xls"):
        df = pd.read_excel(args.csv_path)
    else:
        df = pd.read_csv(args.csv_path)

    print(f"Loaded {len(df)} samples from {args.csv_path}")
    print(f"Anemic (HB < {args.hb_threshold}): {(df[args.hb_col] < args.hb_threshold).sum()}")

    train_df, val_df = train_test_split(
        df, test_size=args.val_split, random_state=args.seed,
        stratify=(df[args.hb_col] < args.hb_threshold).astype(int)
    )

    T_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    T_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_loader = DataLoader(
        EyeHBDataset(train_df, args.train_dataset_path, args.image_col,
                     args.hb_col, args.hb_threshold, T_train),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_works, pin_memory=True)

    val_loader = DataLoader(
        EyeHBDataset(val_df, args.train_dataset_path, args.image_col,
                     args.hb_col, args.hb_threshold, T_val),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_works, pin_memory=True)

    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    # ── Model ────────────────────────────────────────────────────────────────
    model = DSAMambaDualHead(in_chans=3, num_classes=2).to(device)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"DSA-Mamba (official) dual-head | Params: {total/1e6:.2f}M")

    # ── Optim ────────────────────────────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    ce_fn     = nn.CrossEntropyLoss()
    mse_fn    = nn.MSELoss()

    best_val_loss = float("inf")

    # ── Train loop ───────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        bar = tqdm(train_loader, desc=f"[{epoch}/{args.epochs}] Train", leave=False)
        for imgs, labels, hb_true in bar:
            imgs, labels, hb_true = imgs.to(device), labels.to(device), hb_true.to(device)
            optimizer.zero_grad()
            logits, hb_pred = model(imgs)
            loss = args.cls_weight * ce_fn(logits, labels) + args.reg_weight * mse_fn(hb_pred, hb_true)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()
            bar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_train_loss = running_loss / len(train_loader)

        # ── Validate ─────────────────────────────────────────────────────────
        val_loss, acc, auc, mae, rmse, preds, labels_v, hbp, hbt = evaluate(
            model, val_loader, device, ce_fn, mse_fn, args.cls_weight, args.reg_weight)

        print(f"Ep {epoch:3d} | TL: {avg_train_loss:.4f} | VL: {val_loss:.4f} "
              f"| Acc: {acc:.3f} | AUC: {auc:.3f} | MAE: {mae:.2f} g/dL | RMSE: {rmse:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.save_path)
            print(f"  → Saved best model (VL={val_loss:.4f})")

    # ── Final report ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("FINAL EVALUATION (best checkpoint)")
    model.load_state_dict(torch.load(args.save_path, map_location=device))
    _, acc, auc, mae, rmse, preds, labels_v, _, _ = evaluate(
        model, val_loader, device, ce_fn, mse_fn, args.cls_weight, args.reg_weight)

    print(f"Accuracy : {acc:.4f}")
    print(f"AUC      : {auc:.4f}")
    print(f"HB MAE   : {mae:.4f} g/dL")
    print(f"HB RMSE  : {rmse:.4f} g/dL")
    print("\nClassification Report:")
    print(classification_report(labels_v, preds, target_names=["Anemic","Normal"]))


if __name__ == "__main__":
    main()
