"""
eye_hb_model.py  --  07_DSA_Mamba_Custom
Loads the official DSA-Mamba architecture for dual-task HB prediction.

Architecture (First-Ronin/DSA-Mamba):
    Encoder: SS_Conv_SSM stages with patch merging
    Decoder: Cross-attention skip connections + up-sampling
    VSSM(in_depths, out_depths, in_dims, out_dims)

VSSM.forward_backbone(x) -> (B, H', W', C_out)
VSSM.avgpool: AdaptiveAvgPool2d(1)
num_features = out_dims[-1] = 384

cross_attention.py is reconstructed from .pyc bytecode (was missing from repo).
It is written to disk automatically if not already present.

Falls back to MobileNetV3-Small if DSAmamba.py fails to load.
"""

import os, sys, importlib, shutil, torch, torch.nn as nn

_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR  = os.path.join(_THIS_DIR, "model")
_MODELS_DIR = os.path.dirname(_THIS_DIR)

for p in [_MODELS_DIR, _THIS_DIR, _MODEL_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from common.pure_ssm import make_dual_head


# ── cross_attention.py source (reconstructed from .pyc) ─────────────────────
_CROSS_ATTENTION_SRC = '''"""cross_attention.py -- reconstructed from DSA-Mamba .pyc bytecode."""
import torch, torch.nn as nn, torch.nn.functional as F
from torch import Tensor

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=True)
    def forward(self, x):
        B, H, W, C = x.size()
        return self.dwconv(x.permute(0,3,1,2)).flatten(2).transpose(1,2).view(B,H,W,C)

class skip_ffn(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1   = nn.Linear(c1, c2)
        self.dwconv= DWConv(c2)
        self.act   = nn.GELU()
        self.fc2   = nn.Linear(c2, c1)
        self.norm1 = nn.LayerNorm(c2)
        self.norm2 = nn.LayerNorm(c1)
        self.norm3 = nn.LayerNorm(c1)
    def forward(self, x):
        return self.norm3(self.fc2(self.act(self.norm1(self.dwconv(self.fc1(x))))) + x)

class Cross_Attention(nn.Module):
    def __init__(self, key_channels, value_channels, head_count=1):
        super().__init__()
        self.key_channels   = key_channels
        self.head_count     = head_count
        self.value_channels = value_channels
        self.reprojection   = nn.Conv2d(value_channels, key_channels, 1)
        self.norm           = nn.LayerNorm(key_channels)
    def forward(self, x1, x2):
        B, H, W, C = x1.size()
        keys    = x2.view(B, -1, C).transpose(1, 2)
        queries = x1.view(B, -1, C).transpose(1, 2)
        values  = x2.view(B, -1, C).transpose(1, 2)
        hk = self.key_channels   // self.head_count
        hv = self.value_channels // self.head_count
        attended = []
        for i in range(self.head_count):
            k = F.softmax(keys[:,   i*hk:(i+1)*hk, :], dim=2)
            q = F.softmax(queries[:, i*hk:(i+1)*hk, :], dim=1)
            v = values[:, i*hv:(i+1)*hv, :]
            ctx = torch.einsum('bdk,bvk->bdv', k, v)
            attended.append(torch.einsum("bdv,bdl->bvl", ctx, q))
        agg = torch.cat(attended, dim=1).reshape(B, -1, H, W)
        return self.norm(self.reprojection(agg).permute(0, 2, 3, 1))

class CrossAttention(nn.Module):
    def __init__(self, in_dim, key_dim, value_dim, head_count=1, token_mlp=True):
        super().__init__()
        self.norm1    = nn.LayerNorm(in_dim)
        self.linear   = nn.Linear(in_dim, key_dim)
        self.attn     = Cross_Attention(key_dim, value_dim, head_count)
        self.out_proj = nn.Linear(key_dim, in_dim) if key_dim != in_dim else nn.Identity()
        self.norm2    = nn.LayerNorm(in_dim)
        self.mlp      = skip_ffn(in_dim, int(in_dim * 2)) if token_mlp else nn.Identity()
    def forward(self, x1, x2):
        n1   = self.linear(self.norm1(x1))
        n2   = self.linear(self.norm1(x2))
        attn = self.out_proj(self.attn(n1, n2))
        tx   = attn + x1
        return self.mlp(self.norm2(tx)) if isinstance(self.mlp, skip_ffn) else tx
'''


def _ensure_cross_attention():
    """Always write the canonical cross_attention.py from the embedded source.
    This overwrites any stale or incorrect version on disk.
    """
    ca_path = os.path.join(_MODEL_DIR, "cross_attention.py")
    os.makedirs(_MODEL_DIR, exist_ok=True)
    with open(ca_path, "w") as f:
        f.write(_CROSS_ATTENTION_SRC)


def _flush_module_cache():
    """Remove stale Python module cache entries and __pycache__ bytecode
    so that build_model() always loads the freshest code from disk.
    """
    # 1. Evict any previously imported DSA-Mamba sub-modules from sys.modules
    stale = [k for k in sys.modules if any(
        tag in k for tag in ("DSAmamba", "cross_attention", "eye_hb_model",
                             "model.DSAmamba", "model.cross_attention")
    )]
    for k in stale:
        del sys.modules[k]

    # 2. Delete __pycache__ dirs under the model folder so Python recompiles .py → .pyc
    for root, dirs, _ in os.walk(_MODEL_DIR):
        for d in dirs:
            if d == "__pycache__":
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)
    pycache = os.path.join(_THIS_DIR, "__pycache__")
    shutil.rmtree(pycache, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
def build_model(img_size: int = 224):
    """
    Returns DSA-Mamba model with forward(x) -> (logits[B,2], hb_pred[B,1]).
    Always flushes stale module caches and rewrites cross_attention.py so that
    changes to source files are picked up even within a live Kaggle kernel.
    """
    _flush_module_cache()      # purge sys.modules + __pycache__ first
    _ensure_cross_attention()  # (re)write canonical cross_attention.py

    try:
        import model.cross_attention
        import model.DSAmamba
        importlib.reload(model.cross_attention)
        importlib.reload(model.DSAmamba)
        from model.DSAmamba import VSSM as DSA_VSSM

        backbone = DSA_VSSM(
            in_chans=3,
            num_classes=0,
            in_depths=[2, 2, 4],
            out_depths=[2, 2],
            in_dims=[96, 192, 384],
            out_dims=[768, 384],
        )
        feat_dim = backbone.num_features   # out_dims[-1] = 384

        class _DSAWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone  = backbone
                self.avgpool   = nn.AdaptiveAvgPool2d(1)
                self.cls_head  = nn.Linear(feat_dim, 2)
                self.reg_head  = nn.Sequential(
                    nn.Linear(feat_dim, feat_dim // 2), nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(feat_dim // 2, 1))

            def forward(self, x):
                # forward_backbone returns (B, H, W, C) channel-last
                f = self.backbone.forward_backbone(x)
                f = self.avgpool(f.permute(0, 3, 1, 2)).flatten(1)  # (B, feat_dim)
                return self.cls_head(f), self.reg_head(f)

        print(f"  [DSA-Mamba] VSSM loaded from model/DSAmamba.py  (feat_dim={feat_dim})")
        return _DSAWrapper()

    except Exception as e:
        print(f"  [DSA-Mamba] DSAmamba.py load failed ({e})  ->  MobileNetV3-Small fallback.")
        from torchvision.models import mobilenet_v3_small
        bb = mobilenet_v3_small(weights=None)
        bb.classifier[-1] = nn.Identity()
        return make_dual_head(bb, feat_dim=576)
