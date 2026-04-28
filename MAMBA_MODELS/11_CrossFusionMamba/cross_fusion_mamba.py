"""
cross_fusion_mamba.py  --  Cross-Fusion Mamba (CFMamba)
========================================================
Novel architecture for conjunctival pallor-based anemia detection.

Learns FROM the base models:
  ▸ MambaVision : hybrid attention + Mamba outperforms pure SSM
  ▸ MedMamba    : medical images have diagnostically critical local regions
  ▸ DSA-Mamba   : dual-stream processing captures complementary signals
  ▸ Mamba2      : d_state=64 gave the best results on this dataset

IMPORTANT DESIGN DECISION (specific to your dataset):
────────────────────────────────────────────────────────────────────────────────
Your images are already SEGMENTED — only the conjunctival region is visible.
This means:
  ✗ Centre-cropping the image is WRONG. The pallor region IS the entire image.
    Cropping the centre would just give a smaller view of the same conjunctiva.

  ✓ The dual-branch should explore DIFFERENT COLOUR VIEWS of the same region:
      Branch 1 (Global/RGB branch)   → standard RGB view
                                       "What does the conjunctiva look like?"
      Branch 2 (Pallor/Normalised branch) → colour-normalised pallor view
                                       "How pale is the conjunctiva regardless
                                        of illumination?"

PALLOR-NORMALISED BRANCH — what it does and why:
────────────────────────────────────────────────────────────────────────────────
The second branch applies COLOUR NORMALISATION to the image:
    R_norm = R / (R + G + B + eps)
    G_norm = G / (R + G + B + eps)
    B_norm = B / (R + G + B + eps)

Why this helps:
  Problem:  Camera exposure and lighting vary between patients.
            A brightly lit pale conjunctiva may have similar RAW R values
            to a dimly lit healthy conjunctiva.
  Solution: Dividing by total intensity removes the overall brightness,
            leaving only the RELATIVE colour proportions.
  Effect:
    Healthy (pink)  → R_norm ≈ 0.45, G_norm ≈ 0.30, B_norm ≈ 0.25
    Anemic  (pale)  → R_norm ≈ G_norm ≈ B_norm ≈ 0.33  (no colour dominance)
  The model now clearly sees pallor as a convergence of the 3 normalised
  channels towards 1/3, regardless of whether the image was bright or dark.

CROSS-ATTENTION:
  Once Branch 1 (RGB) and Branch 2 (pallor-normalised) have each been processed
  by their own SSM towers, bidirectional cross-attention fuses them:
    • Branch 1 QUERIES into Branch 2: "where does the pallor-view confirm pallor?"
    • Branch 2 QUERIES into Branch 1: "what does the raw image look like there?"
  This lets the model verify pallor from TWO independent viewpoints before deciding.

Usage
-----
    from cross_fusion_mamba import build_cross_fusion_mamba
    model = build_cross_fusion_mamba(img_size=224, embed_dim=128, depth=4)
    logits, hb = model(images)   # images: (B, 3, H, W)  RGB normalised
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# ── Locate common/pure_ssm.py ────────────────────────────────────────────────
def _add_common():
    here   = os.path.dirname(os.path.abspath(__file__))
    models = os.path.dirname(here)
    for p in [models, os.path.join(models, "common")]:
        if p not in sys.path:
            sys.path.insert(0, p)

_add_common()

try:
    from common.pure_ssm import _PureMamba
except ImportError:
    from pure_ssm import _PureMamba

try:
    from timm.models.layers import trunc_normal_
except ImportError:
    def trunc_normal_(t, std=0.02):
        nn.init.trunc_normal_(t, std=std)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Pallor-Normalised View
#    (replaces the centre-crop from the original version)
# ═══════════════════════════════════════════════════════════════════════════════

def pallor_normalised_view(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Convert RGB image to colour-normalised pallor view.
    Input:  (B, 3, H, W) — standard normalised RGB (values approx in [-1, 1])
    Output: (B, 3, H, W) — each channel divided by total (R+G+B), centred

    The normalisation removes luminance variation, leaving only colour ratios.
    An anemic conjunctiva will have R_norm ≈ G_norm ≈ B_norm ≈ 1/3 (no colour).
    A healthy conjunctiva will have R_norm >> G_norm (reddish).

    Why not use CIELAB?
      CIELAB requires cv2 and is not differentiable through the model.
      This normalisation achieves a similar illumination-invariant effect
      purely in PyTorch, and is differentiable (useful for future grad-CAM work).
    """
    # Shift from ~[-1,1] to ~[0,1] for meaningful colour ratios
    x_pos  = (x + 1.0) / 2.0                          # (B,3,H,W), range [0,1]
    total  = x_pos.sum(dim=1, keepdim=True) + eps      # (B,1,H,W)
    x_norm = x_pos / total                             # (B,3,H,W), each ch in [0,1], sum=1
    # Re-centre: mean across pixels should be ~0.33 for each ch
    # Subtract channel mean to make it zero-centred for the model
    x_norm = x_norm - x_norm.mean(dim=(2, 3), keepdim=True)
    return x_norm


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Single SSM Branch
# ═══════════════════════════════════════════════════════════════════════════════

class SSMBlock(nn.Module):
    """Pre-norm _PureMamba block with residual."""
    def __init__(self, embed_dim: int, d_state: int, d_conv: int, expand: int):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.ssm  = _PureMamba(embed_dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ssm(self.norm(x))


class SSMBranch(nn.Module):
    """
    Patch-embed → SSM tower.
    Used separately for the RGB branch and the pallor-normalised branch.
    """
    def __init__(self, embed_dim: int, patch_size: int, depth: int,
                 d_state: int = 64, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, stride=patch_size)
        self.blocks = nn.ModuleList([
            SSMBlock(embed_dim, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.norm(tokens)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Bidirectional Cross-Attention Fusion
# ═══════════════════════════════════════════════════════════════════════════════

class CrossAttentionFusion(nn.Module):
    """
    Bidirectional cross-attention between RGB tokens (a) and pallor-norm tokens (b).

    a_enhanced = a + Attention(Q=a, K=b, V=b)
                   "RGB branch queries pallor-normalised context"
    b_enhanced = b + Attention(Q=b, K=a, V=a)
                   "Pallor branch queries raw RGB context"

    Learnable scalar gates (gate_a, gate_b) control how much each branch
    is updated — the model can learn to suppress one branch if it's unhelpful.
    """
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        while num_heads > 1 and dim % num_heads != 0:
            num_heads -= 1
        self.cross_ab = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.cross_ba = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_a   = nn.LayerNorm(dim)
        self.norm_b   = nn.LayerNorm(dim)
        self.gate_a   = nn.Parameter(torch.ones(1) * 0.5)
        self.gate_b   = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        ab_out, _ = self.cross_ab(a, b, b)
        ba_out, _ = self.cross_ba(b, a, a)
        a_enh = self.norm_a(a + self.gate_a.clamp(0, 1) * ab_out)
        b_enh = self.norm_b(b + self.gate_b.clamp(0, 1) * ba_out)
        return a_enh, b_enh


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Fusion Refinement
# ═══════════════════════════════════════════════════════════════════════════════

class FusionRefinement(nn.Module):
    """Additional SSM blocks run on the fused token sequence."""
    def __init__(self, embed_dim: int, depth: int = 2,
                 d_state: int = 64, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([
            SSMBlock(embed_dim, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Main model
# ═══════════════════════════════════════════════════════════════════════════════

class CrossFusionMamba(nn.Module):
    """
    Cross-Fusion Mamba (CFMamba) for conjunctival anemia detection.

    Architecture:
      Input RGB ─┬─ (as-is, standard RGB)       → SSMBranch_rgb    → tok_rgb  (B,N,D)
                 └─ pallor_normalised_view(x)    → SSMBranch_pallor → tok_p    (B,N,D)
                          ↓
                   CrossAttentionFusion (bidirectional)
                          ↓
                   Cat [tok_rgb, tok_p] → Linear(2D→D) → FusionRefinement
                          ↓
                   Append CLS → Dual Head

    forward(x: Tensor[B, 3, H, W]) → (logits[B, 2], hb_pred[B, 1])
    """

    def __init__(self,
                 img_size:     int   = 224,
                 patch_size:   int   = 16,
                 embed_dim:    int   = 128,
                 depth:        int   = 4,
                 fusion_depth: int   = 2,
                 n_heads:      int   = 4,
                 d_state:      int   = 64,    # ← 64 matches best Mamba2
                 d_conv:       int   = 4,
                 expand:       int   = 2):
        super().__init__()
        n_patches = (img_size // patch_size) ** 2

        # Branch 1: standard RGB
        self.rgb_branch    = SSMBranch(embed_dim, patch_size, depth,
                                        d_state=d_state, d_conv=d_conv, expand=expand)
        # Branch 2: pallor-normalised view of the SAME segmented image
        self.pallor_branch = SSMBranch(embed_dim, patch_size, depth,
                                        d_state=d_state, d_conv=d_conv, expand=expand)

        # Separate positional embeddings for each branch
        self.pos_rgb    = nn.Parameter(torch.zeros(1, n_patches, embed_dim))
        self.pos_pallor = nn.Parameter(torch.zeros(1, n_patches, embed_dim))
        trunc_normal_(self.pos_rgb,    std=0.02)
        trunc_normal_(self.pos_pallor, std=0.02)

        # Bidirectional cross-attention fusion
        self.cross_fusion = CrossAttentionFusion(embed_dim, num_heads=n_heads)

        # Project concatenated (2D) back to D, then refine
        self.fusion_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.fusion_norm = nn.LayerNorm(embed_dim)
        self.refinement  = FusionRefinement(embed_dim, depth=fusion_depth,
                                             d_state=d_state, d_conv=d_conv, expand=expand)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token, std=0.02)

        # Dual head
        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(embed_dim // 2, 2))
        self.reg_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(embed_dim // 2, 1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        B = x.shape[0]

        # ── Branch 1: standard RGB ────────────────────────────────────────────
        tok_rgb    = self.rgb_branch(x)    + self.pos_rgb      # (B, N, D)

        # ── Branch 2: pallor-normalised (illumination-invariant) ─────────────
        x_pallor   = pallor_normalised_view(x)
        tok_pallor = self.pallor_branch(x_pallor) + self.pos_pallor  # (B, N, D)

        # ── Cross-attention ───────────────────────────────────────────────────
        tok_rgb, tok_pallor = self.cross_fusion(tok_rgb, tok_pallor)

        # ── Concatenate, project back, refine ─────────────────────────────────
        fused  = torch.cat([tok_rgb, tok_pallor], dim=-1)      # (B, N, 2D)
        fused  = self.fusion_norm(self.fusion_proj(fused))      # (B, N, D)
        fused  = self.refinement(fused)                         # (B, N, D)

        # ── CLS pool ──────────────────────────────────────────────────────────
        cls    = self.cls_token.expand(B, -1, -1)
        fused  = torch.cat([fused, cls], dim=1)                 # (B, N+1, D)
        cls_out = fused[:, -1]

        return self.cls_head(cls_out), self.reg_head(cls_out)

    def extra_repr(self):
        p = sum(v.numel() for v in self.parameters())
        return (f"params={p/1e6:.2f}M  "
                f"branch_1=RGB  branch_2=pallor_normalised")


# ═══════════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════════

def build_cross_fusion_mamba(img_size: int = 224,
                              embed_dim: int = 128,
                              depth: int = 4,
                              d_state: int = 64,
                              **kwargs) -> CrossFusionMamba:
    """
    Build CrossFusionMamba.

    Key parameters:
        img_size  : input image size (default 224)
        embed_dim : token embedding dimension (default 128)
        depth     : SSM layers per branch (default 4)
        d_state   : SSM state dim — 64 matches best Mamba2 results (default 64)

    The dual-branch design:
        Branch 1 = standard RGB      (raw colour context)
        Branch 2 = pallor-normalised (illumination-invariant pallor view)
    Both are 224×224 — no cropping is used since images are already segmented.
    """
    return CrossFusionMamba(img_size=img_size, embed_dim=embed_dim,
                             depth=depth, d_state=d_state, **kwargs)


if __name__ == "__main__":
    m = build_cross_fusion_mamba(d_state=64)
    x = torch.randn(2, 3, 224, 224)
    l, h = m(x)
    print(f"CFMamba  logits:{l.shape}  hb:{h.shape}  {m.extra_repr()}")

    # Show what pallor_normalised_view does
    pv = pallor_normalised_view(x)
    print(f"Pallor view range: [{pv.min():.3f}, {pv.max():.3f}]  "
          f"channel means: {pv.mean((0,2,3)).tolist()}")
