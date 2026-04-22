"""
eye_hb_model.py  --  06_Mamba3_Minimal
Builds a vision model using the pure-PyTorch Mamba3 blocks from mamba3.py.

mamba3.py is fully hardware-agnostic (no CUDA ops, no triton).
We stack Mamba3 SSM blocks inside a standard patch-embed vision encoder:
    PatchEmbed (Conv2d) -> N x [RMSNorm + Mamba3Block] -> mean-pool -> dual head

Mamba3 block signature: forward(u, h=None) -> (y, h_cache)
    u: (B, L, D)   y: (B, L, D)
"""

import os, sys, torch, torch.nn as nn

_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.dirname(_THIS_DIR)
for p in [_MODELS_DIR, _THIS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from common.pure_ssm import _PureMamba, _VisionMambaDual

try:
    from timm.models.layers import trunc_normal_
except ImportError:
    def trunc_normal_(t, std=0.02): nn.init.trunc_normal_(t, std=std)


def build_model(img_size: int = 224, embed_dim: int = 128, depth: int = 4):
    """
    Returns Mamba3 vision model with forward(x) -> (logits[B,2], hb_pred[B,1]).
    """
    mamba3_py = os.path.join(_THIS_DIR, "mamba3.py")

    try:
        import importlib.util
        spec     = importlib.util.spec_from_file_location("_mamba3_mod", mamba3_py)
        m3_mod   = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m3_mod)

        Mamba3       = m3_mod.Mamba3
        Mamba3Config = m3_mod.Mamba3Config

        # Validate headdim divides d_inner = expand * d_model
        headdim = max(16, embed_dim // 2)
        assert (embed_dim * 2) % headdim == 0, \
            f"headdim={headdim} must divide d_inner={embed_dim*2}"

        class Mamba3Block(nn.Module):
            """Thin wrapper so Mamba3 matches the ssm_fn(d_model) convention."""
            def __init__(self, d_model: int):
                super().__init__()
                h   = max(16, d_model // 2)
                cfg = Mamba3Config(
                    d_model=d_model, n_layer=1, d_state=32,
                    expand=2, headdim=h, chunk_size=64, use_mimo=False)
                self.ssm  = Mamba3(cfg)
                # NOTE: no self.norm here — _VisionMambaDual's pre-norm already
                # normalises the input before calling this block (double-norm bug removed)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Mamba3.forward(u, h=None) returns (y, cache)
                y, _ = self.ssm(x)   # x is already layer-normed by _VisionMambaDual
                return y             # residual added by _VisionMambaDual

        ssm_fn = lambda d: Mamba3Block(d)
        print(f"  [Mamba3] pure-PyTorch Mamba3 blocks loaded from mamba3.py")

    except Exception as e:
        ssm_fn = lambda d: _PureMamba(d, d_state=32, expand=2)
        print(f"  [Mamba3] mamba3.py failed ({e})  ->  pure-PyTorch SSM approximation.")

    return _VisionMambaDual(ssm_fn, img_size=img_size, embed_dim=embed_dim, depth=depth)
