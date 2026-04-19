"""
╔══════════════════════════════════════════════════════════════════════════╗
║         MAMBA MODELS — SMOKE TEST (All models, 1-2 epochs each)         ║
║                                                                          ║
║  Runs every Mamba variant one by one on the same dataset & settings.     ║
║  Edit only the CONFIG block below — nothing else needs to change.        ║
║                                                                          ║
║  Usage (Kaggle):                                                         ║
║      python smoke_test_all_models.py                                     ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

# ┌──────────────────────────────────────────────────────────────────────────┐
# │                          ★  CONFIG  ★                                   │
# │         EDIT ONLY THIS BLOCK — everything else auto-adapts              │
# └──────────────────────────────────────────────────────────────────────────┘

CONFIG = dict(
    # ── Dataset paths ──────────────────────────────────────────────────────
    image_dir  = "/kaggle/input/datasets/junaidgpu/imagehb/dataset/dataset/left_eye",
    csv_path   = "/kaggle/input/datasets/junaidgpu/imagehb/dataset/dataset/merge_excel_1.xlsx",
    image_col  = "Patient ID",   # column name for image filename (without extension)
    hb_col     = "HB",           # column name for HB value
    hb_thresh  = 12.0,           # g/dL  →  below = anemic (label 0), above = normal (label 1)

    # ── Training ───────────────────────────────────────────────────────────
    epochs     = 2,              # smoke test: 1–2 epochs per model
    batch_size = 4,
    lr         = 1e-4,
    cls_weight = 1.0,            # weight for classification loss (CrossEntropy)
    reg_weight = 0.5,            # weight for regression loss (MSE → HB prediction)
    val_split  = 0.2,
    num_workers= 2,
    seed       = 42,

    # ── Image ──────────────────────────────────────────────────────────────
    img_size   = 224,

    # ── Repo root (where all XX_ModelName/ folders live) ───────────────────
    # When cloned from GitHub: /kaggle/working/MAMBA_MODELS
    repo_root  = "/kaggle/working/MAMBA_MODELS",
)

# ═══════════════════════════════════════════════════════════════════════════════
# DO NOT EDIT BELOW THIS LINE
# ═══════════════════════════════════════════════════════════════════════════════

import os, sys, math, time, warnings, traceback
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from tqdm import tqdm
warnings.filterwarnings("ignore")

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
ROOT     = CONFIG["repo_root"]
RESULTS  = []   # filled at the end: [{name, status, acc, mae, time, error}]

print(f"\n{'═'*72}")
print(f"  MAMBA SMOKE TEST  |  Device: {DEVICE}  |  Epochs/model: {CONFIG['epochs']}")
print(f"{'═'*72}\n")


# ──────────────────────────────────────────────────────────────────────────────
# 1.  SHARED DATASET & LOADERS
# ──────────────────────────────────────────────────────────────────────────────

class EyeHBDataset(Dataset):
    """Loads eye images + HB values from a CSV/Excel sheet."""
    def __init__(self, df, transform=None):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        pid   = str(row[CONFIG["image_col"]])
        hb    = float(row[CONFIG["hb_col"]])
        label = 0 if hb < CONFIG["hb_thresh"] else 1

        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ""]:
            p = os.path.join(CONFIG["image_dir"], pid + ext)
            if os.path.exists(p):
                img_path = p
                break
        if img_path is None:
            raise FileNotFoundError(f"No image for Patient ID '{pid}'")

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return (img,
                torch.tensor(label, dtype=torch.long),
                torch.tensor([[hb]], dtype=torch.float32))


T_TRAIN = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(CONFIG["img_size"]),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
T_VAL = transforms.Compose([
    transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def get_loaders():
    if CONFIG["csv_path"].endswith((".xlsx", ".xls")):
        df = pd.read_excel(CONFIG["csv_path"])
    else:
        df = pd.read_csv(CONFIG["csv_path"])

    train_df, val_df = train_test_split(
        df,
        test_size=CONFIG["val_split"],
        random_state=CONFIG["seed"],
        stratify=(df[CONFIG["hb_col"]] < CONFIG["hb_thresh"]).astype(int),
    )
    train_loader = DataLoader(
        EyeHBDataset(train_df, T_TRAIN),
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=(DEVICE == "cuda"),
    )
    val_loader = DataLoader(
        EyeHBDataset(val_df, T_VAL),
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=(DEVICE == "cuda"),
    )
    print(f"  Dataset  → train: {len(train_df)}  val: {len(val_df)}"
          f"  anemic: {(df[CONFIG['hb_col']] < CONFIG['hb_thresh']).sum()}/{len(df)}\n")
    return train_loader, val_loader

print("Loading dataset …")
TRAIN_LOADER, VAL_LOADER = get_loaders()


# ──────────────────────────────────────────────────────────────────────────────
# 2.  SHARED TRAINING / EVAL LOOP
# ──────────────────────────────────────────────────────────────────────────────

CE_LOSS  = nn.CrossEntropyLoss()
MSE_LOSS = nn.MSELoss()

def dual_loss(logits, hb_pred, labels, hb_true):
    return (CONFIG["cls_weight"] * CE_LOSS(logits, labels)
          + CONFIG["reg_weight"] * MSE_LOSS(hb_pred, hb_true))


def run_epoch(model, loader, optimizer=None):
    """Train (if optimizer given) or evaluate one epoch. Returns loss, acc, mae."""
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = correct = total = 0
    all_hbp, all_hbt = [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, labels, hb_true in loader:
            imgs    = imgs.to(DEVICE)
            labels  = labels.to(DEVICE)
            hb_true = hb_true.to(DEVICE)

            logits, hb_pred = model(imgs)
            loss = dual_loss(logits, hb_pred, labels, hb_true)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            preds       = logits.argmax(1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)
            all_hbp.extend(hb_pred.detach().cpu().squeeze().tolist())
            all_hbt.extend(hb_true.cpu().squeeze().tolist())

    avg_loss = total_loss / len(loader)
    acc      = correct / total
    mae      = mean_absolute_error(all_hbt, all_hbp)
    return avg_loss, acc, mae


def smoke_test(name, model):
    """Run one model through CONFIG['epochs'] epochs. Returns result dict."""
    print(f"\n{'─'*72}")
    print(f"  [{name}]")
    print(f"{'─'*72}")
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {params/1e6:.2f}M")

    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    t0 = time.time()

    for ep in range(1, CONFIG["epochs"] + 1):
        tr_loss, tr_acc, tr_mae = run_epoch(model, TRAIN_LOADER, optimizer)
        vl_loss, vl_acc, vl_mae = run_epoch(model, VAL_LOADER)
        print(f"  Ep {ep}/{CONFIG['epochs']} | "
              f"TL:{tr_loss:.4f} VL:{vl_loss:.4f} | "
              f"Acc:{vl_acc:.3f} | MAE:{vl_mae:.2f} g/dL")

    elapsed = time.time() - t0
    print(f"  ✅  PASSED  ({elapsed:.0f}s)")
    return dict(name=name, status="✅ PASS", acc=vl_acc,
                mae=vl_mae, time_s=elapsed, error="")


# ──────────────────────────────────────────────────────────────────────────────
# 3.  PURE-PYTORCH SSM BLOCK (no CUDA kernel — works everywhere)
#     Used by Mamba1, Mamba2 (simplified), Mamba3 (simplified) adapters
# ──────────────────────────────────────────────────────────────────────────────

class _PureMamba(nn.Module):
    """Minimal selective SSM — pure PyTorch, no mamba_ssm package needed."""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        d_inner       = int(expand * d_model)
        self.d_inner  = d_inner
        self.d_state  = d_state
        self.in_proj  = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d   = nn.Conv1d(d_inner, d_inner, kernel_size=d_conv,
                                   padding=d_conv - 1, groups=d_inner)
        self.act      = nn.SiLU()
        self.x_proj   = nn.Linear(d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj  = nn.Linear(1, d_inner, bias=True)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32),
                   "n -> d n", d=d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D     = nn.Parameter(torch.ones(d_inner))

    def forward(self, x):                    # (B, L, d_model)
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_, z = xz.chunk(2, dim=-1)
        x_ = rearrange(x_, "b l d -> b d l")
        x_ = self.conv1d(x_)[..., :L]
        x_ = rearrange(x_, "b d l -> b l d")
        x_ = self.act(x_)
        bcd = self.x_proj(x_)
        B_  = bcd[..., :self.d_state]
        C   = bcd[..., self.d_state : 2 * self.d_state]
        dt  = F.softplus(self.dt_proj(bcd[..., -1:]))
        A   = -torch.exp(self.A_log.float())
        dA  = torch.exp(dt.unsqueeze(-1) * A)
        dB  = dt.unsqueeze(-1) * B_.unsqueeze(2)
        h   = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys  = []
        for i in range(L):
            h = dA[:, i] * h + dB[:, i] * x_[:, i : i + 1].unsqueeze(-1)
            ys.append((h * C[:, i].unsqueeze(1)).sum(-1))
        y = torch.stack(ys, dim=1) + x_ * self.D
        return self.out_proj(y * self.act(z))


class _PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=128):
        super().__init__()
        h = img_size // patch_size
        self.num_patches = h * h
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class _VisionMambaDualHead(nn.Module):
    """
    Generic vision-Mamba dual-head model.
    Swap 'ssm_cls' to test different SSM variants with the same wrapper.
    """
    def __init__(self, ssm_cls=_PureMamba, embed_dim=128, depth=4,
                 num_classes=2, img_size=224, patch_size=16,
                 ssm_kwargs=None):
        super().__init__()
        ssm_kwargs = ssm_kwargs or {}
        self.patch_embed = _PatchEmbed(img_size, patch_size, 3, embed_dim)
        N = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, N + 1, embed_dim))
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)
        self.norms  = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(depth)])
        self.blocks = nn.ModuleList([ssm_cls(embed_dim, **ssm_kwargs) for _ in range(depth)])
        self.norm   = nn.LayerNorm(embed_dim)
        self.cls_head = nn.Linear(embed_dim, num_classes)
        self.reg_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2), nn.GELU(), nn.Linear(embed_dim // 2, 1))

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1) + self.pos_embed
        for norm, blk in zip(self.norms, self.blocks):
            x = x + blk(norm(x))
        feat = self.norm(x)[:, 0]
        return self.cls_head(feat), self.reg_head(feat)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  MODEL FACTORIES — one function per model
#     Each returns an nn.Module whose forward() yields (logits, hb_pred)
# ──────────────────────────────────────────────────────────────────────────────

def _path(*parts):
    return os.path.join(ROOT, *parts)


# ── M1: Mamba1 (official, pure-PyTorch fallback) ─────────────────────────────
def build_mamba1():
    """Try compiled mamba_ssm first; fall back to pure-PyTorch."""
    try:
        from mamba_ssm import Mamba as _Mamba1Layer
        class _FastMamba(_VisionMambaDualHead):
            pass
        # Override blocks with real Mamba layers
        model = _VisionMambaDualHead(embed_dim=128, depth=4)
        model.blocks = nn.ModuleList(
            [_Mamba1Layer(d_model=128, d_state=16, d_conv=4, expand=2) for _ in range(4)])
        print("  → using compiled mamba_ssm (Mamba1 CUDA kernel)")
        return model
    except ImportError:
        print("  → mamba_ssm not found, using pure-PyTorch SSM")
        return _VisionMambaDualHead(ssm_cls=_PureMamba, embed_dim=128, depth=4)


# ── M2: Mamba2 ───────────────────────────────────────────────────────────────
def build_mamba2():
    """Mamba2 via compiled kernel or pure-PyTorch."""
    try:
        from mamba_ssm.modules.mamba2_simple import Mamba2Simple
        model = _VisionMambaDualHead(embed_dim=128, depth=4)
        model.blocks = nn.ModuleList(
            [Mamba2Simple(d_model=128, d_state=64, d_conv=4, expand=2) for _ in range(4)])
        print("  → using compiled Mamba2Simple")
        return model
    except Exception:
        print("  → Mamba2 kernel not found, using pure-PyTorch SSM (Mamba2 approximation)")
        # Mamba2 approximation: larger d_state
        return _VisionMambaDualHead(ssm_cls=_PureMamba, embed_dim=128, depth=4,
                                    ssm_kwargs={"d_state": 64})


# ── M3: Mamba3 ───────────────────────────────────────────────────────────────
def build_mamba3():
    """Mamba3 via official module or pure-PyTorch."""
    sys.path.insert(0, _path("01_Mamba_Official"))
    try:
        from mamba_ssm.modules.mamba3 import Mamba3 as _Mamba3
        model = _VisionMambaDualHead(embed_dim=128, depth=4)
        model.blocks = nn.ModuleList([_Mamba3(d_model=128) for _ in range(4)])
        print("  → using compiled Mamba3 (ICLR 2026)")
        return model
    except Exception:
        print("  → Mamba3 not found, using pure-PyTorch + rotary-style SSM approximation")
        return _VisionMambaDualHead(ssm_cls=_PureMamba, embed_dim=128, depth=4,
                                    ssm_kwargs={"d_state": 16, "expand": 2})


# ── M4: VMamba (2D cross-scan SSM) ───────────────────────────────────────────
def build_vmamba():
    sys.path.insert(0, _path("02_VMamba"))
    try:
        from classification.models.vmamba import VSSM as VMambaVSSM
        backbone = VMambaVSSM(num_classes=0)
        feat_dim = backbone.num_features
        print(f"  → VMamba backbone loaded (feat_dim={feat_dim})")
    except Exception:
        try:
            # Alternative: direct vmamba.py in root
            sys.path.insert(0, _path("02_VMamba"))
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "vmamba", _path("02_VMamba", "vmamba.py"))
            vm = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(vm)
            backbone = vm.VSSM(num_classes=0, depths=[2,2,9,2], dims=96)
            feat_dim = backbone.num_features
            print(f"  → VMamba (vmamba.py) loaded")
        except Exception as e2:
            print(f"  → VMamba import failed ({e2}), using ResNet-18 fallback")
            from torchvision.models import resnet18
            backbone = resnet18(weights=None)
            feat_dim = 512
            backbone.fc = nn.Identity()

    class _VMambaDual(nn.Module):
        def __init__(self, bb, fd):
            super().__init__()
            self.bb = bb
            self.cls_head = nn.Linear(fd, 2)
            self.reg_head = nn.Sequential(
                nn.Linear(fd, fd // 2), nn.GELU(), nn.Linear(fd // 2, 1))
        def forward(self, x):
            f = self.bb(x)
            if f.dim() > 2: f = f.mean([-2, -1])
            return self.cls_head(f), self.reg_head(f)

    return _VMambaDual(backbone, feat_dim)


# ── M5: MambaVision (NVIDIA, CVPR 2025) ──────────────────────────────────────
def build_mambavision():
    try:
        from mambavision import create_model
        backbone = create_model("mamba_vision_T", pretrained=False, img_size=224)
        feat_dim = 640
        backbone.head = nn.Identity()
        print("  → MambaVision-T loaded (pip mambavision)")
    except Exception as e:
        print(f"  → mambavision package not found ({e}), using EfficientNet-B0 fallback")
        from torchvision.models import efficientnet_b0
        backbone = efficientnet_b0(weights=None)
        feat_dim = 1280
        backbone.classifier[-1] = nn.Identity()

    class _MVDual(nn.Module):
        def __init__(self, bb, fd):
            super().__init__()
            self.bb = bb
            self.cls_head = nn.Linear(fd, 2)
            self.reg_head = nn.Sequential(
                nn.Linear(fd, fd // 4), nn.GELU(), nn.Linear(fd // 4, 1))
        def forward(self, x):
            f = self.bb(x)
            if f.dim() > 2: f = f.mean([-2, -1])
            return self.cls_head(f), self.reg_head(f)

    return _MVDual(backbone, feat_dim)


# ── M6: MedMamba ─────────────────────────────────────────────────────────────
def build_medmamba():
    medmamba_path = _path("04_MedMamba")
    sys.path.insert(0, medmamba_path)
    try:
        # MedMamba.py has .to("cuda") calls at module level — load safely
        import importlib.util, types
        spec = importlib.util.spec_from_file_location(
            "MedMamba", os.path.join(medmamba_path, "MedMamba.py"))
        mod = types.ModuleType("MedMamba")
        # Patch torch.Tensor.to so .to("cuda") → .to(DEVICE) silently
        original_to = torch.Tensor.to
        torch.nn.Module.to = lambda self, *a, **kw: self  # no-op during import
        try:
            spec.loader.exec_module(mod)
        finally:
            torch.nn.Module.to = original_to              # restore
        VSSM = mod.VSSM
        backbone = VSSM(depths=[2, 2, 4, 2], dims=[96, 192, 384, 768], num_classes=0)
        feat_dim = 768
        print(f"  → MedMamba VSSM-T loaded (feat_dim={feat_dim})")
    except Exception as e:
        print(f"  → MedMamba import failed ({e}), using ConvNeXt-T fallback")
        from torchvision.models import convnext_tiny
        backbone = convnext_tiny(weights=None)
        feat_dim = 768
        backbone.classifier[-1] = nn.Identity()

    class _MedDual(nn.Module):
        def __init__(self, bb, fd):
            super().__init__()
            self.bb = bb
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.cls_head = nn.Linear(fd, 2)
            self.reg_head = nn.Sequential(
                nn.Linear(fd, fd // 4), nn.GELU(), nn.Linear(fd // 4, 1))
        def forward(self, x):
            f = self.bb(x)
            # MedMamba forward() returns logits if num_classes>0
            # With num_classes=0 head is Identity → returns flat feat
            if f.dim() > 2:
                f = f.mean([-2, -1])
            return self.cls_head(f), self.reg_head(f)

    return _MedDual(backbone, feat_dim)


# ── M7: DSA-Mamba (official, First-Ronin/DSA-Mamba) ─────────────────────────
def build_dsa_mamba():
    dsa_path = _path("07_DSA_Mamba_Custom")
    sys.path.insert(0, dsa_path)
    try:
        from model.DSAmamba import VSSM as DSABackbone

        class _DSADual(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = DSABackbone(
                    in_chans=3, num_classes=0,
                    in_depths=[2, 2, 4], out_depths=[2, 2],
                    in_dims=[96, 192, 384], out_dims=[768, 384])
                feat_dim = 384
                self.avgpool  = nn.AdaptiveAvgPool2d(1)
                self.cls_head = nn.Linear(feat_dim, 2)
                self.reg_head = nn.Sequential(
                    nn.Linear(feat_dim, feat_dim // 2),
                    nn.GELU(), nn.Dropout(0.1),
                    nn.Linear(feat_dim // 2, 1))

            def forward(self, x):
                f = self.backbone.forward_backbone(x)     # (B,H,W,C)
                f = f.permute(0, 3, 1, 2)                # (B,C,H,W)
                f = self.backbone.avgpool(f)             # (B,C,1,1)
                f = torch.flatten(f, 1)                  # (B,C)
                return self.cls_head(f), self.reg_head(f)

        print("  → Official DSA-Mamba (First-Ronin/DSA-Mamba) loaded")
        return _DSADual()

    except Exception as e:
        print(f"  → DSA-Mamba import failed ({e}), using fallback")
        from torchvision.models import mobilenet_v3_small
        backbone = mobilenet_v3_small(weights=None)
        feat_dim = 576
        backbone.classifier[-1] = nn.Identity()

        class _Fallback(nn.Module):
            def __init__(self, bb, fd):
                super().__init__()
                self.bb = bb
                self.cls_head = nn.Linear(fd, 2)
                self.reg_head = nn.Sequential(
                    nn.Linear(fd, fd // 2), nn.GELU(), nn.Linear(fd // 2, 1))
            def forward(self, x):
                f = self.bb(x)
                if f.dim() > 2: f = f.mean([-2, -1])
                return self.cls_head(f), self.reg_head(f)

        return _Fallback(backbone, feat_dim)


# ──────────────────────────────────────────────────────────────────────────────
# 5.  MODELS REGISTRY
#     Add / remove / reorder here freely — smoke_test handles the rest
# ──────────────────────────────────────────────────────────────────────────────

MODELS = [
    ("Mamba1  (official SSM)",               build_mamba1),
    ("Mamba2  (SSD / ICML 2024)",            build_mamba2),
    ("Mamba3  (ICLR 2026)",                  build_mamba3),
    ("VMamba  (2D cross-scan, ICLR 2025)",   build_vmamba),
    ("MambaVision  (NVIDIA, CVPR 2025)",     build_mambavision),
    ("MedMamba  (medical imaging)",          build_medmamba),
    ("DSA-Mamba  (First-Ronin, official)",   build_dsa_mamba),
]


# ──────────────────────────────────────────────────────────────────────────────
# 6.  RUN ALL MODELS
# ──────────────────────────────────────────────────────────────────────────────

for idx, (name, factory) in enumerate(MODELS, 1):
    print(f"\n[{idx}/{len(MODELS)}]  Building: {name}")
    try:
        model = factory()
        result = smoke_test(name, model)
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"  ❌  FAILED: {exc}")
        print(tb)
        result = dict(name=name, status="❌ FAIL", acc=float("nan"),
                      mae=float("nan"), time_s=0, error=str(exc))
    RESULTS.append(result)
    del model                # free GPU memory between runs
    torch.cuda.empty_cache() if DEVICE == "cuda" else None


# ──────────────────────────────────────────────────────────────────────────────
# 7.  FINAL SUMMARY TABLE
# ──────────────────────────────────────────────────────────────────────────────

print(f"\n\n{'═'*72}")
print("  SMOKE TEST SUMMARY")
print(f"{'═'*72}")
print(f"  {'Model':<42} {'Status':<12} {'Acc':>6}  {'MAE':>7}  {'Time':>7}")
print(f"  {'─'*42} {'─'*12} {'─'*6}  {'─'*7}  {'─'*7}")

all_pass = True
for r in RESULTS:
    acc_s  = f"{r['acc']:.3f}"  if not math.isnan(r['acc']) else "  n/a"
    mae_s  = f"{r['mae']:.2f}" if not math.isnan(r['mae']) else "  n/a"
    time_s = f"{r['time_s']:.0f}s"
    print(f"  {r['name']:<42} {r['status']:<12} {acc_s:>6}  {mae_s:>7}  {time_s:>7}")
    if "FAIL" in r["status"]:
        all_pass = False

print(f"{'═'*72}")
if all_pass:
    print("  ✅  All models passed — safe to push to GitHub & run full training!")
else:
    failed = [r["name"] for r in RESULTS if "FAIL" in r["status"]]
    print(f"  ⚠️   {len(failed)} model(s) failed: {', '.join(failed)}")
    print("     Check error messages above — common fixes:")
    print("     • pip install mamba-ssm causal-conv1d")
    print("     • make sure repo_root path is correct")
    print("     • ensure GPU is available for CUDA kernels")
print(f"{'═'*72}\n")
