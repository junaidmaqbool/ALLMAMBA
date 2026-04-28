"""
moe_mamba.py  --  Mixture-of-Experts Mamba (MoEMamba)
=======================================================
Novel architecture for conjunctival pallor-based anemia detection.

Learns FROM the base models:
  > EfficientMamba : pretrained CNN backbones give rich features
  > MedMamba       : channel-level processing is critical in medical images
  > Mamba2         : d_state=64 gave the best results on this dataset

Novel contributions:
  1. Channel Experts: separate SSM blocks, each trained on one colour/clinical channel.
  2. Top-2 Gating: per-token routing to 2 most relevant experts.
  3. Pallor-Index Expert (optional, N_EXPERTS=5).
  4. d_state=64 matching the best-performing Mamba2 configuration.

UNDERSTANDING THE EXPERTS (N_EXPERTS parameter)
=================================================
Your images: segmented conjunctival photographs. Clinical pallor signal:
  Healthy: R >> G (reddish-pink conjunctiva)
  Anemic:  R ~= G (pale/white, no colour dominance)

  Expert 0 - Red channel (R)
    Haemoglobin gives the conjunctiva its red colour. Low HB -> less red.
    Strongest single-channel predictor of pallor.

  Expert 1 - Green channel (G)
    Reference/contrast channel. Healthy: low G relative to R.
    Anemic: G ~= R (both channels equal -> pale).

  Expert 2 - Blue channel (B)
    Least discriminative for anemia. Acts as control channel.

  Expert 3 - Luminance (Y = 0.299R + 0.587G + 0.114B)
    Overall brightness. Pale tissue is often BRIGHTER (more reflective).
    Captures brightness-based pallor independent of colour.

  Expert 4 - Pallor Index = (R-G)/(R+G+eps)  [N_EXPERTS=5 only]
    Clinical pallor measure. Range [-1, 1]:
      PI > 0  -> R > G -> reddish -> NORMAL haemoglobin
      PI ~= 0 -> R ~= G -> pale   -> ANEMIC
    Dedicated expert for borderline samples (HB ~= 12 g/dL).
    Set N_EXPERTS=4 for R,G,B,Lum only.
    Set N_EXPERTS=5 to add the Pallor Index expert.

GATING NETWORK
==============
For each token (image patch):
  1. Linear: D-dim embedding -> N_EXPERTS scores
  2. Softmax: scores -> probabilities
  3. Top-2: select 2 best experts, weighted by their probabilities
Result: token-level routing. Pale patches -> PallorIndex/Red experts.

Usage
-----
    from moe_mamba import build_moe_mamba
    model = build_moe_mamba(img_size=224, embed_dim=128, depth=4,
                             n_experts=5, d_state=64)
    logits, hb = model(images)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

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


# ======================================================================
# 1. Top-2 Gating
# ======================================================================

class Top2Gate(nn.Module):
    """
    Per-token Top-2 gating over n_experts.
    Returns indices (B,N,2) and normalised weights (B,N,2) summing to 1.
    The gating LEARNS which experts matter for which tokens.
    """
    def __init__(self, embed_dim: int, n_experts: int):
        super().__init__()
        self.n_experts = n_experts
        self.gate_proj = nn.Linear(embed_dim, n_experts, bias=False)

    def forward(self, x: torch.Tensor):
        scores  = F.softmax(self.gate_proj(x), dim=-1)
        top2_w, top2_idx = torch.topk(scores, k=2, dim=-1)
        top2_w  = top2_w / top2_w.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        return top2_idx, top2_w


# ======================================================================
# 2. Channel Tokeniser
# ======================================================================

class ChannelTokeniser(nn.Module):
    """
    Extracts per-expert input channels from the RGB image.
    n_experts = 4: [R, G, B, Luminance]
    n_experts = 5: [R, G, B, Luminance, PallorIndex]
    """
    EXPERT_NAMES = ["Red (R)", "Green (G)", "Blue (B)", "Luminance (Y)",
                    "Pallor Index (R-G)/(R+G)"]

    def __init__(self, embed_dim: int, patch_size: int = 16, n_experts: int = 5):
        super().__init__()
        assert n_experts in (4, 5), "n_experts must be 4 or 5"
        self.n_experts  = n_experts
        self.embeds = nn.ModuleList([
            nn.Conv2d(1, embed_dim, patch_size, stride=patch_size)
            for _ in range(n_experts)
        ])
        self.rgb_embed = nn.Conv2d(3, embed_dim, patch_size, stride=patch_size)
        self.norms     = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(n_experts)])
        print(f"  MoEMamba experts ({n_experts}): {self.EXPERT_NAMES[:n_experts]}")

    def _extract_channels(self, x: torch.Tensor) -> list:
        R, G, B = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        eps = 1e-6
        channels = [
            R,
            G,
            B,
            0.299 * R + 0.587 * G + 0.114 * B,
        ]
        if self.n_experts == 5:
            pallor_idx = (R - G) / (R + G + eps)
            channels.append(pallor_idx)
        return channels

    def forward(self, x: torch.Tensor):
        channels = self._extract_channels(x)
        per_ch   = []
        for i, ch in enumerate(channels):
            tok = self.embeds[i](ch).flatten(2).transpose(1, 2)
            per_ch.append(self.norms[i](tok))
        base_tok = self.rgb_embed(x).flatten(2).transpose(1, 2)
        return base_tok, per_ch


# ======================================================================
# 3. Channel Injection Layer
# ======================================================================

class ChannelInjection(nn.Module):
    """
    Injects per-channel expert knowledge into the base RGB token sequence.
    Learnable alpha weights control each channel's contribution.
    """
    def __init__(self, embed_dim: int, n_experts: int, d_state: int = 64):
        super().__init__()
        self.ssms  = nn.ModuleList([
            _PureMamba(embed_dim, d_state=d_state)
            for _ in range(n_experts)
        ])
        self.alpha = nn.Parameter(torch.ones(n_experts) / n_experts)
        self.norm  = nn.LayerNorm(embed_dim)

    def forward(self, base: torch.Tensor, per_ch: list) -> torch.Tensor:
        w = F.softmax(self.alpha, dim=0)
        injection = sum(
            w[i] * self.ssms[i](self.norm(per_ch[i]))
            for i in range(len(per_ch))
        )
        return base + injection


# ======================================================================
# 4. MoE SSM Block
# ======================================================================

class MoESSMBlock(nn.Module):
    """
    MoE block: Top-2 expert routing + residual + FFN.
    d_state=64 matches the best Mamba2 configuration.
    """
    def __init__(self, embed_dim: int, n_experts: int = 5,
                 d_state: int = 64, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.n_experts = n_experts
        self.gate   = Top2Gate(embed_dim, n_experts)
        self.experts = nn.ModuleList([
            _PureMamba(embed_dim, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_experts)
        ])
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2), nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim))
        self.drop  = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        xn = self.norm1(x)
        top2_idx, top2_w = self.gate(xn)
        expert_outs = torch.stack([e(xn) for e in self.experts], dim=2)
        idx_exp  = top2_idx.unsqueeze(-1).expand(-1, -1, -1, x.shape[-1])
        gathered = expert_outs.gather(2, idx_exp)
        fused    = (gathered * top2_w.unsqueeze(-1)).sum(dim=2)
        x = residual + self.drop(fused)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


# ======================================================================
# 5. Main model
# ======================================================================

class MoEMamba(nn.Module):
    """
    Mixture-of-Experts Mamba for conjunctival pallor detection.
    forward(x: Tensor[B, 3, H, W]) -> (logits[B, 2], hb_pred[B, 1])
    """

    def __init__(self,
                 img_size:   int = 224,
                 patch_size: int = 16,
                 embed_dim:  int = 128,
                 depth:      int = 4,
                 n_experts:  int = 5,
                 d_state:    int = 64,
                 d_conv:     int = 4,
                 expand:     int = 2):
        super().__init__()
        self.n_experts = n_experts
        n_patches = (img_size // patch_size) ** 2

        self.tokeniser = ChannelTokeniser(embed_dim, patch_size, n_experts)
        self.ch_inject = ChannelInjection(embed_dim, n_experts, d_state=d_state)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            MoESSMBlock(embed_dim, n_experts=n_experts,
                        d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

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
        base_tok, per_ch = self.tokeniser(x)
        tokens  = self.ch_inject(base_tok, per_ch)
        cls     = self.cls_token.expand(B, -1, -1)
        tokens  = torch.cat([tokens, cls], dim=1) + self.pos_embed
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens  = self.norm(tokens)
        cls_out = tokens[:, -1]
        return self.cls_head(cls_out), self.reg_head(cls_out)

    def extra_repr(self):
        p = sum(v.numel() for v in self.parameters())
        experts = ChannelTokeniser.EXPERT_NAMES[:self.n_experts]
        return f"params={p/1e6:.2f}M  experts={experts}"


# ======================================================================
# Factory
# ======================================================================

def build_moe_mamba(img_size: int = 224,
                    embed_dim: int = 128,
                    depth: int = 4,
                    n_experts: int = 5,
                    d_state: int = 64,
                    **kwargs) -> MoEMamba:
    """
    Build MoEMamba.

    Key parameters:
        img_size  : input image size (default 224)
        embed_dim : token embedding dimension (default 128)
        depth     : number of MoESSMBlock layers (default 4)
        n_experts : 4 = (R,G,B,Lum)  |  5 = (R,G,B,Lum,PallorIndex)
        d_state   : SSM state dim - 64 matches best Mamba2 results (default 64)
    """
    return MoEMamba(img_size=img_size, embed_dim=embed_dim, depth=depth,
                    n_experts=n_experts, d_state=d_state, **kwargs)


if __name__ == "__main__":
    for ne in (4, 5):
        m = build_moe_mamba(n_experts=ne, d_state=64)
        x = torch.randn(2, 3, 224, 224)
        l, h = m(x)
        print(f"n_experts={ne}  logits:{l.shape}  hb:{h.shape}  {m.extra_repr()}")
