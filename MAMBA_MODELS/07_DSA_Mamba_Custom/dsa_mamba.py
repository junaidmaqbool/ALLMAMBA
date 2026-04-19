"""
DSA-Mamba: Dense Spectral Attention Mamba
==========================================
Custom implementation for retinal eye image analysis.

Tasks:
  - Binary Classification : Anemic (HB < threshold) vs Non-Anemic
  - Regression            : Predict exact HB value (g/dL)

Architecture:
  Eye Image → Patch Embed → DSA-Mamba Blocks → [CLS token / GAP]
                                             ├─► Classification Head
                                             └─► Regression Head

The "Dense Spectral Attention" mechanism adds a channel-wise
spectral attention on top of Mamba's selective SSM, allowing the
model to weight RGB/spectral channels differently per patch — which
matters for hemoglobin estimation from conjunctival images.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


# ─────────────────────────────────────────────────────────────────────────────
# 1. Lightweight Mamba-style SSM (pure PyTorch, no CUDA kernel dependency)
#    Falls back gracefully when mamba_ssm is not installed on Kaggle.
# ─────────────────────────────────────────────────────────────────────────────

class MambaSSM(nn.Module):
    """
    Minimal selective SSM (Mamba inner loop).
    Pure-PyTorch fallback — no CUDA kernel required.
    Works on Kaggle out of the box.
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model  = d_model
        self.d_state  = d_state
        self.d_inner  = int(expand * d_model)

        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d   = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                   padding=d_conv - 1, groups=self.d_inner, bias=True)
        self.act      = nn.SiLU()

        # SSM parameters
        self.x_proj   = nn.Linear(self.d_inner, self.d_state * 2 + 1, bias=False)  # B, C, dt
        self.dt_proj  = nn.Linear(1, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Log-space A initialisation
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32), 'n -> d n', d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D     = nn.Parameter(torch.ones(self.d_inner))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: x  (B, L, d_model)
        Returns: (B, L, d_model)
        """
        B, L, _ = x.shape

        xz   = self.in_proj(x)                  # (B, L, 2*d_inner)
        x_, z = xz.chunk(2, dim=-1)              # each (B, L, d_inner)

        # Conv over sequence
        x_   = rearrange(x_, 'b l d -> b d l')
        x_   = self.conv1d(x_)[..., :L]
        x_   = rearrange(x_, 'b d l -> b l d')
        x_   = self.act(x_)

        # SSM params
        bcd  = self.x_proj(x_)                  # (B, L, d_state*2 + 1)
        B_   = bcd[..., :self.d_state]           # (B, L, d_state)
        C    = bcd[..., self.d_state:2*self.d_state]
        dt   = bcd[..., -1:]                     # (B, L, 1)
        dt   = F.softplus(self.dt_proj(dt))      # (B, L, d_inner)

        A    = -torch.exp(self.A_log.float())    # (d_inner, d_state)

        # Simplified discrete SSM scan (cumulative)
        dA   = torch.exp(dt.unsqueeze(-1) * A)  # (B, L, d_inner, d_state)
        dB   = dt.unsqueeze(-1) * B_.unsqueeze(2)  # (B, L, d_inner, d_state)

        h    = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys   = []
        for i in range(L):
            h = dA[:, i] * h + dB[:, i] * x_[:, i:i+1].unsqueeze(-1)
            y = (h * C[:, i].unsqueeze(1)).sum(-1)  # (B, d_inner)
            ys.append(y)
        y    = torch.stack(ys, dim=1)            # (B, L, d_inner)

        y    = y + x_ * self.D
        y    = y * self.act(z)
        return self.out_proj(y)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Dense Spectral Attention (DSA) module
#    Channel-wise attention computed in frequency domain via FFT.
# ─────────────────────────────────────────────────────────────────────────────

class DenseSpectralAttention(nn.Module):
    """
    Applies channel attention using spectral (frequency) energy cues.
    Motivation: HB correlates with specific wavelength absorption in
    conjunctival images; spectral channel weighting captures this.
    """
    def __init__(self, d_model: int, reduction: int = 4):
        super().__init__()
        d_mid = max(d_model // reduction, 16)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_mid),
            nn.GELU(),
            nn.Linear(d_mid, d_model),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: x  (B, L, d_model)
        Returns: (B, L, d_model)  — channel-scaled
        """
        # Spectral energy per channel: mean of |FFT|^2 across sequence
        X_f    = torch.fft.rfft(x, dim=1)                # (B, L//2+1, d_model)
        energy = (X_f.abs() ** 2).mean(dim=1)            # (B, d_model)
        energy = energy / (energy.sum(dim=-1, keepdim=True) + 1e-8)
        scale  = self.fc(energy).unsqueeze(1)            # (B, 1, d_model)
        return x * scale


# ─────────────────────────────────────────────────────────────────────────────
# 3. DSA-Mamba Block = DSA → LayerNorm → MambaSSM → residual
# ─────────────────────────────────────────────────────────────────────────────

class DSAMambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.dsa   = DenseSpectralAttention(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ssm   = MambaSSM(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # DSA branch
        x = x + self.drop_path(self.dsa(self.norm1(x)))
        # SSM branch
        x = x + self.drop_path(self.ssm(self.norm2(x)))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 4. Patch Embedding (Vision ViT-style)
# ─────────────────────────────────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_chans: int = 3, embed_dim: int = 256):
        super().__init__()
        img_size   = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                         # (B, C, H', W')
        x = x.flatten(2).transpose(1, 2)         # (B, N, C)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 5. Full DSA-Mamba Model
# ─────────────────────────────────────────────────────────────────────────────

class DSAMamba(nn.Module):
    """
    DSA-Mamba for retinal image analysis.

    Outputs:
        logits      : (B, num_classes)  — for cross-entropy classification
        hb_pred     : (B, 1)            — for HB regression (g/dL)
    """
    def __init__(
        self,
        img_size:    int   = 224,
        patch_size:  int   = 16,
        in_chans:    int   = 3,
        num_classes: int   = 2,        # 0=anemic, 1=non-anemic
        embed_dim:   int   = 256,
        depth:       int   = 6,
        d_state:     int   = 16,
        d_conv:      int   = 4,
        expand:      int   = 2,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches      = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            DSAMambaBlock(embed_dim, d_state=d_state, d_conv=d_conv,
                          expand=expand, drop_path=dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Dual heads
        self.cls_head = nn.Linear(embed_dim, num_classes)
        self.reg_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x : (B, 3, H, W)  eye image
        Returns:
            logits  : (B, num_classes)
            hb_pred : (B, 1)
        """
        B = x.shape[0]
        x = self.patch_embed(x)                          # (B, N, C)

        cls = self.cls_token.expand(B, -1, -1)           # (B, 1, C)
        x   = torch.cat([cls, x], dim=1)                 # (B, N+1, C)
        x   = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_feat = x[:, 0]                               # CLS token

        logits  = self.cls_head(cls_feat)                # (B, num_classes)
        hb_pred = self.reg_head(cls_feat)                # (B, 1)
        return logits, hb_pred


# ─────────────────────────────────────────────────────────────────────────────
# 6. Dual-task loss
# ─────────────────────────────────────────────────────────────────────────────

class DSAMambaLoss(nn.Module):
    """
    Combined classification + regression loss.

    total = cls_weight * CE(logits, labels)
          + reg_weight * MSE(hb_pred, hb_true)
    """
    def __init__(self, cls_weight: float = 1.0, reg_weight: float = 1.0):
        super().__init__()
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.ce_loss    = nn.CrossEntropyLoss()
        self.mse_loss   = nn.MSELoss()

    def forward(self, logits, hb_pred, labels, hb_true):
        """
        logits   : (B, C)
        hb_pred  : (B, 1)
        labels   : (B,)   long — 0/1
        hb_true  : (B, 1) float — actual HB values
        """
        cls_l = self.ce_loss(logits, labels)
        reg_l = self.mse_loss(hb_pred, hb_true)
        return self.cls_weight * cls_l + self.reg_weight * reg_l, cls_l, reg_l


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model  = DSAMamba(img_size=224, patch_size=16, embed_dim=128, depth=4)
    x      = torch.randn(2, 3, 224, 224)
    logits, hb = model(x)
    print(f"Classification output : {logits.shape}")   # (2, 2)
    print(f"HB regression output  : {hb.shape}")       # (2, 1)

    criterion  = DSAMambaLoss()
    labels     = torch.tensor([0, 1])
    hb_true    = torch.tensor([[10.5], [13.2]])
    loss, cl, rl = criterion(logits, hb, labels, hb_true)
    print(f"Total loss={loss:.4f}  cls={cl:.4f}  reg={rl:.4f}")
