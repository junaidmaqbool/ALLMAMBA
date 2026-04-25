"""
pure_ssm.py  --  Pure-PyTorch selective SSM and dual-head vision wrappers.

Used as a fallback when compiled CUDA kernels (mamba_ssm, triton) are absent,
AND as a shared helper imported by every eye_hb_model.py.

Exports
-------
_PureMamba(d_model, d_state, d_conv, expand)
    Single-block selective SSM in pure PyTorch.

_VisionMambaDual(ssm_fn, img_size, embed_dim, depth, patch_size)
    Patch-embed -> SSM tower -> CLS token -> cls head + reg head.
    forward(x) -> (logits[B,2], hb_pred[B,1])

make_dual_head(backbone, feat_dim)
    Attach dual cls+reg head to any backbone whose forward returns (B, feat_dim).
    forward(x) -> (logits[B,2], hb_pred[B,1])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

try:
    from timm.models.layers import trunc_normal_
except ImportError:
    def trunc_normal_(tensor, std=0.02):
        nn.init.trunc_normal_(tensor, std=std)


# ─────────────────────────────────────────────────────────────────────────────
class _PureMamba(nn.Module):
    """Pure-PyTorch selective SSM (no CUDA kernel required)."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        d_inner = int(expand * d_model)
        self.d_inner = d_inner
        self.d_state = d_state

        self.in_proj  = nn.Linear(d_model,  d_inner * 2, bias=False)
        self.conv1d   = nn.Conv1d(d_inner, d_inner, d_conv,
                                  padding=d_conv - 1, groups=d_inner)
        self.act      = nn.SiLU()
        self.x_proj   = nn.Linear(d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj  = nn.Linear(1, d_inner, bias=True)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32),
                   "n -> d n", d=d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D     = nn.Parameter(torch.ones(d_inner))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D)  ->  (B, L, D)"""
        # AMP guard: disable autocast so ALL ops (including nn.Linear) run in fp32.
        # Without this, linear projections output fp16 while A_log parameters stay fp32,
        # causing "GET was unable to find an engine" on dt*A (fp16 × fp32).
        orig_dtype = x.dtype
        with torch.cuda.amp.autocast(enabled=False):
            x = x.float()
            B, L, _ = x.shape
            xz = self.in_proj(x)
            x_, z = xz.chunk(2, dim=-1)

            x_ = rearrange(x_, "b l d -> b d l")
            x_ = self.conv1d(x_)[..., :L]
            x_ = rearrange(x_, "b d l -> b l d")
            x_ = self.act(x_)

            bcd = self.x_proj(x_)
            B_  = bcd[..., : self.d_state]
            C   = bcd[..., self.d_state : 2 * self.d_state]
            dt  = F.softplus(self.dt_proj(bcd[..., -1:]))

            A  = -torch.exp(self.A_log.float())          # fp32
            dA = torch.exp(dt.unsqueeze(-1) * A)         # fp32 × fp32 — no mixed-dtype error
            dB = dt.unsqueeze(-1) * B_.unsqueeze(2)

            h  = torch.zeros(B, self.d_inner, self.d_state,
                             device=x.device, dtype=torch.float32)
            ys = []
            for i in range(L):
                h = dA[:, i] * h + dB[:, i] * x_[:, i].unsqueeze(-1)
                ys.append((h * C[:, i].unsqueeze(1)).sum(-1))

            y   = torch.stack(ys, dim=1) + x_ * self.D
            out = self.out_proj(y * self.act(z))
        return out.to(orig_dtype)   # restore fp16/bf16 for the rest of the network


# ─────────────────────────────────────────────────────────────────────────────
class _VisionMambaDual(nn.Module):
    """
    Minimal vision backbone using SSM blocks.
    PatchEmbed -> SSM tower with CLS token -> dual head.
    """

    def __init__(self, ssm_fn, img_size: int = 224,
                 embed_dim: int = 128, depth: int = 4, patch_size: int = 16):
        super().__init__()
        n_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, stride=patch_size)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed   = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)

        self.norms  = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(depth)])
        self.blocks = nn.ModuleList([ssm_fn(embed_dim) for _ in range(depth)])
        self.norm   = nn.LayerNorm(embed_dim)

        self.cls_head = nn.Linear(embed_dim, 2)
        self.reg_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2), nn.GELU(),
            nn.Linear(embed_dim // 2, 1))

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)           # (B, N, C)
        # CLS appended LAST so the causal SSM can attend to all patches before CLS
        x = torch.cat([x, self.cls_token.expand(B, -1, -1)], dim=1) # (B, N+1, C)
        x = x + self.pos_embed

        for norm, blk in zip(self.norms, self.blocks):
            x = x + blk(norm(x))

        feat = self.norm(x)[:, -1]  # CLS token (last position sees all patches)
        return self.cls_head(feat), self.reg_head(feat)


# ─────────────────────────────────────────────────────────────────────────────
def make_dual_head(backbone: nn.Module, feat_dim: int) -> nn.Module:
    """
    Wrap a backbone that returns (B, feat_dim) features with dual heads.
    Handles 4-D spatial (B,C,H,W), 4-D BHWC, 3-D (B,L,C), and 2-D (B,C).
    Returns a module whose forward(x) -> (logits[B,2], hb_pred[B,1]).
    """
    class _DualHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone  = backbone
            self.pool      = nn.AdaptiveAvgPool2d(1)
            self.cls_head  = nn.Linear(feat_dim, 2)
            self.reg_head  = nn.Sequential(
                nn.Linear(feat_dim, feat_dim // 4), nn.GELU(),
                nn.Linear(feat_dim // 4, 1))

        def _flatten(self, f: torch.Tensor) -> torch.Tensor:
            if isinstance(f, (tuple, list)):
                f = f[0]
            if f.dim() == 4:
                # Distinguish (B,C,H,W) from (B,H,W,C)
                if f.shape[1] == feat_dim:          # channel-first
                    f = self.pool(f).flatten(1)
                else:                               # channel-last
                    f = f.mean([1, 2])
            elif f.dim() == 3:
                f = f.mean(1)                       # (B,L,C) -> (B,C)
            return f

        def forward(self, x: torch.Tensor):
            f = self.backbone(x)
            f = self._flatten(f)
            return self.cls_head(f), self.reg_head(f)

    return _DualHead()
