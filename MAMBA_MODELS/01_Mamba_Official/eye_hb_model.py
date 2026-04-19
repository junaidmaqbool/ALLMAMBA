"""
eye_hb_model.py  --  01_Mamba_Official
Builds Mamba1, Mamba2, Mamba3 vision models for dual-task HB prediction.

All three share the same _VisionMambaDual wrapper:
    PatchEmbed -> N x SSM-block -> CLS token -> (cls_head, reg_head)

With mamba-ssm installed (Kaggle T4):
    Mamba1  -> mamba_ssm.Mamba
    Mamba2  -> mamba_ssm.modules.mamba2_simple.Mamba2Simple
    Mamba3  -> loaded from 06_Mamba3_Minimal/mamba3.py (pure PyTorch)

Without mamba-ssm:
    All three fall back to _PureMamba (pure PyTorch selective SSM).
"""

import os, sys, torch, torch.nn as nn

# ── ensure common/ is importable ─────────────────────────────────────────────
_MODELS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MODELS_DIR not in sys.path:
    sys.path.insert(0, _MODELS_DIR)

from common.pure_ssm import _PureMamba, _VisionMambaDual


# ─────────────────────────────────────────────────────────────────────────────
def build_mamba1(img_size: int = 224, embed_dim: int = 128, depth: int = 4):
    """Returns a Mamba1 vision model (logits, hb_pred)."""
    try:
        from mamba_ssm import Mamba
        ssm_fn = lambda d: Mamba(d_model=d, d_state=16, d_conv=4, expand=2)
        print("  [Mamba1] compiled mamba_ssm kernel loaded.")
    except Exception as e:
        ssm_fn = lambda d: _PureMamba(d, d_state=16)
        print(f"  [Mamba1] mamba_ssm not available ({e}) -> pure-PyTorch fallback.")
    return _VisionMambaDual(ssm_fn, img_size=img_size, embed_dim=embed_dim, depth=depth)


def build_mamba2(img_size: int = 224, embed_dim: int = 128, depth: int = 4):
    """Returns a Mamba2/SSD vision model (logits, hb_pred)."""
    try:
        from mamba_ssm.modules.mamba2_simple import Mamba2Simple
        ssm_fn = lambda d: Mamba2Simple(d_model=d, d_state=64, d_conv=4, expand=2)
        print("  [Mamba2] compiled Mamba2Simple kernel loaded.")
    except Exception as e:
        ssm_fn = lambda d: _PureMamba(d, d_state=64)
        print(f"  [Mamba2] Mamba2Simple not available ({e}) -> pure-PyTorch fallback (d_state=64).")
    return _VisionMambaDual(ssm_fn, img_size=img_size, embed_dim=embed_dim, depth=depth)


def build_mamba3(img_size: int = 224, embed_dim: int = 128, depth: int = 4):
    """
    Returns a Mamba3 vision model (logits, hb_pred).
    Uses pure-PyTorch Mamba3 block from 06_Mamba3_Minimal/mamba3.py.
    Falls back to _PureMamba if that also fails.
    """
    mamba3_dir = os.path.join(_MODELS_DIR, "06_Mamba3_Minimal")
    if mamba3_dir not in sys.path:
        sys.path.insert(0, mamba3_dir)

    try:
        from mamba3 import Mamba3, Mamba3Config

        # Build a tiny config that fits embed_dim
        # headdim must divide d_inner = expand*d_model; use headdim=embed_dim//2
        headdim = max(16, embed_dim // 2)
        cfg = Mamba3Config(
            d_model=embed_dim,
            n_layer=1,        # we stack depth blocks manually
            d_state=32,       # must be even
            expand=2,
            headdim=headdim,
            chunk_size=1,
        )

        class _Mamba3Block(nn.Module):
            """Wraps a single Mamba3 SSM so it matches ssm_fn(d_model) -> (B,L,D) interface."""
            def __init__(self, d_model: int):
                super().__init__()
                h = max(16, d_model // 2)
                c = Mamba3Config(d_model=d_model, n_layer=1, d_state=32,
                                 expand=2, headdim=h, chunk_size=1)
                self.ssm = Mamba3(c)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Mamba3.forward(u, h=None) -> (B, L, D)
                y, _ = self.ssm(x)
                return y

        ssm_fn = lambda d: _Mamba3Block(d)
        print("  [Mamba3] pure-PyTorch Mamba3 block loaded from mamba3.py.")
    except Exception as e:
        ssm_fn = lambda d: _PureMamba(d, d_state=32, expand=2)
        print(f"  [Mamba3] mamba3.py load failed ({e}) -> pure-PyTorch approximation.")

    return _VisionMambaDual(ssm_fn, img_size=img_size, embed_dim=embed_dim, depth=depth)
