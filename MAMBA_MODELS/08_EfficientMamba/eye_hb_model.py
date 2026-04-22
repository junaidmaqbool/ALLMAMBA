"""
eye_hb_model.py — 08_EfficientMamba
=====================================
Pretrained EfficientNet-B0 backbone  +  Mamba SSM sequence blocks.

WHY THIS IS FUNDAMENTALLY DIFFERENT FROM ALL OTHER MODELS HERE
--------------------------------------------------------------
Every other model in this repo starts from RANDOM weights on <2000 images.
A Mamba SSM (or any deep network) trained from scratch on this little data
will memorise the mean HB and stop — the feature extractor never becomes
meaningful because there isn't enough data to bootstrap it.

This model breaks that cycle by separating two concerns:

  1. EfficientNet-B0 (pretrained on 1.28M ImageNet images)
       → already knows edges, textures, colour gradients, vessels.
       → Frozen by default: its weights don't change, zero overfitting risk.
       → Outputs a rich spatial feature map (B, 1280, 7, 7) = 49 patch tokens.

  2. Mamba SSM blocks (trained from scratch on our data)
       → Receives those 49 ImageNet-quality tokens as a sequence.
       → Learns WHICH spatial relationships matter for conjunctival pallor.
       → Much smaller parameter count → converges with less data.

ARCHITECTURE
  Input (B, 3, 224, 224)
      ↓
  EfficientNet-B0 features  →  (B, 1280, 7, 7)
      ↓  flatten + linear projection
  Token sequence             →  (B, 49, embed_dim)
      ↓  positional encoding + N × Mamba SSM blocks
  Global mean pool           →  (B, embed_dim)
      ↓
  cls_head (B, 2)   +   reg_head (B, 1)

USAGE
  from eye_hb_model import build_model
  model = build_model(img_size=224, embed_dim=256, depth=4, freeze_backbone=True)

FREEZE_BACKBONE
  True  (default) — only Mamba blocks + heads are trained.
                    ~1.5 M trainable params.  Fast convergence.  Recommended first run.
  False           — full fine-tune.  All 5.3 M params train.
                    Use once classification/regression heads converge.
"""

import os
import sys
import torch
import torch.nn as nn

# ── make common/ importable ───────────────────────────────────────────────────
_MODELS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MODELS_DIR not in sys.path:
    sys.path.insert(0, _MODELS_DIR)

from common.pure_ssm import _PureMamba

try:
    from timm.models.layers import trunc_normal_
except ImportError:
    def trunc_normal_(t, std=0.02):
        nn.init.trunc_normal_(t, std=std)


# ─────────────────────────────────────────────────────────────────────────────
class _SSMTower(nn.Module):
    """
    Small stack of Mamba SSM blocks that processes a token sequence.
    Each block is pre-norm: x = x + SSM(LayerNorm(x))
    """
    def __init__(self, ssm_fn, embed_dim: int, depth: int):
        super().__init__()
        self.norms  = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(depth)])
        self.blocks = nn.ModuleList([ssm_fn(embed_dim) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for norm, blk in zip(self.norms, self.blocks):
            x = x + blk(norm(x))
        return self.out_norm(x)


# ─────────────────────────────────────────────────────────────────────────────
class EfficientMamba(nn.Module):
    """
    EfficientNet-B0 (pretrained) backbone + Mamba SSM token processor.
    """

    def __init__(
        self,
        ssm_fn,
        embed_dim:       int  = 256,
        depth:           int  = 4,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        # ── 1. EfficientNet-B0 backbone ───────────────────────────────────
        try:
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            print("  [EfficientMamba] EfficientNet-B0 loaded (ImageNet pretrained).")
        except (ImportError, AttributeError):
            # Older torchvision (<= 0.11)
            from torchvision.models import efficientnet_b0
            backbone = efficientnet_b0(pretrained=True)
            print("  [EfficientMamba] EfficientNet-B0 loaded (old torchvision API).")

        # `backbone.features` is the conv stack — output: (B, 1280, 7, 7) @ 224px
        self.feature_extractor = backbone.features
        in_channels = 1280
        self.n_tokens = 7 * 7  # 49 tokens for 224×224 input

        if freeze_backbone:
            for p in self.feature_extractor.parameters():
                p.requires_grad = False
            print("  [EfficientMamba] Backbone FROZEN  "
                  "(Mamba + heads only, ~1.5M trainable params).")
        else:
            print("  [EfficientMamba] Backbone UNFROZEN  "
                  "(full fine-tune, ~5.3M trainable params).")

        # ── 2. Projection: 1280 → embed_dim ──────────────────────────────
        self.proj = nn.Sequential(
            nn.Linear(in_channels, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_tokens, embed_dim))
        trunc_normal_(self.pos_embed, std=0.02)

        # ── 3. Mamba SSM tower ────────────────────────────────────────────
        self.ssm_tower = _SSMTower(ssm_fn, embed_dim, depth)

        # ── 4. Dual heads ─────────────────────────────────────────────────
        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim // 2, 2),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor):
        # x: (B, 3, 224, 224)
        feat   = self.feature_extractor(x)             # (B, 1280, 7, 7)
        tokens = feat.flatten(2).transpose(1, 2)        # (B, 49, 1280)
        tokens = self.proj(tokens) + self.pos_embed     # (B, 49, embed_dim)
        tokens = self.ssm_tower(tokens)                 # (B, 49, embed_dim)
        pooled = tokens.mean(dim=1)                     # (B, embed_dim)
        return self.cls_head(pooled), self.reg_head(pooled)


# ─────────────────────────────────────────────────────────────────────────────
def build_model(
    img_size:        int  = 224,
    embed_dim:       int  = 256,
    depth:           int  = 4,
    freeze_backbone: bool = True,
) -> EfficientMamba:
    """
    Build and return the EfficientMamba model.

    Parameters
    ----------
    img_size        : image size fed to the model (should be 224 for best EfficientNet fit)
    embed_dim       : SSM token dimension after projection (256 = good balance)
    depth           : number of Mamba SSM blocks (4 recommended)
    freeze_backbone : True = fast convergence; False = full fine-tune after warm-up
    """
    # Try compiled Mamba1 kernel first (best quality)
    try:
        from mamba_ssm import Mamba
        ssm_fn = lambda d: Mamba(d_model=d, d_state=16, d_conv=4, expand=2)
        print("  [EfficientMamba] Using compiled mamba_ssm (Mamba1 CUDA kernel).")
    except Exception as e:
        ssm_fn = lambda d: _PureMamba(d, d_state=16)
        print(f"  [EfficientMamba] mamba_ssm unavailable ({e})"
              " → pure-PyTorch SSM fallback.")

    return EfficientMamba(
        ssm_fn,
        embed_dim=embed_dim,
        depth=depth,
        freeze_backbone=freeze_backbone,
    )
