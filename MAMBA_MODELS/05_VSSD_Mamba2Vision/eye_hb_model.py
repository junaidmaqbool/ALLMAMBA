"""
eye_hb_model.py  --  05_VSSD_Mamba2Vision
Loads VMAMBA2 from classification/models/mamba2.py for dual-task HB prediction.

VSSD is a Mamba-2 vision architecture (ICCV 2025).
VMAMBA2(num_classes=0) -> (B, num_features) features.
Default: embed_dim=64, depths=4 -> num_features = 64 * 2^3 = 512.

The model needs mamba_util.py (PatchMerging, Stem, etc.) from the same folder.
Falls back to a pure-PyTorch VisionMambaDual if import fails.
"""

import os, sys, torch, torch.nn as nn

_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.dirname(_THIS_DIR)
_CLS_DIR    = os.path.join(_THIS_DIR, "classification")
_MOD_DIR    = os.path.join(_CLS_DIR, "models")

for p in [_MODELS_DIR, _THIS_DIR, _CLS_DIR, _MOD_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from common.pure_ssm import make_dual_head, _PureMamba, _VisionMambaDual


def build_model(img_size: int = 224):
    """
    Returns VSSD/VMAMBA2 model with forward(x) -> (logits[B,2], hb_pred[B,1]).
    """
    try:
        import importlib.util

        mamba2_py  = os.path.join(_MOD_DIR, "mamba2.py")
        util_py    = os.path.join(_MOD_DIR, "mamba_util.py")

        # Load mamba_util first (required by mamba2.py)
        spec_util  = importlib.util.spec_from_file_location("mamba_util", util_py)
        mod_util   = importlib.util.module_from_spec(spec_util)
        sys.modules["mamba_util"] = mod_util
        spec_util.loader.exec_module(mod_util)

        # Load mamba2 (VSSD/VMAMBA2)
        spec_m2   = importlib.util.spec_from_file_location("_vssd_mamba2", mamba2_py)
        mod_m2    = importlib.util.module_from_spec(spec_m2)
        spec_m2.loader.exec_module(mod_m2)

        VMAMBA2  = mod_m2.VMAMBA2

        # Tiny configuration
        backbone = VMAMBA2(
            img_size=img_size,
            num_classes=0,
            embed_dim=64,
            depths=[2, 4, 8, 2],
        )
        feat_dim = backbone.num_features   # 64 * 2^3 = 512

        class _VSSDWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone  = backbone
                self.pool      = nn.AdaptiveAvgPool2d(1)
                self.cls_head  = nn.Linear(feat_dim, 2)
                self.reg_head  = nn.Sequential(
                    nn.Linear(feat_dim, feat_dim // 4), nn.GELU(),
                    nn.Linear(feat_dim // 4, 1))

            def forward(self, x):
                f = self.backbone(x)   # (B, feat_dim) when num_classes=0
                if f.dim() == 4:
                    f = self.pool(f).flatten(1) if f.shape[1] == feat_dim else f.mean([1, 2])
                elif f.dim() == 3:
                    f = f.mean(1)
                return self.cls_head(f), self.reg_head(f)

        print(f"  [VSSD] VMAMBA2 loaded from classification/models/mamba2.py  (feat_dim={feat_dim})")
        return _VSSDWrapper()

    except Exception as e:
        print(f"  [VSSD] VMAMBA2 load failed ({e})  ->  pure-PyTorch SSM fallback.")
        ssm_fn = lambda d: _PureMamba(d, d_state=64)
        return _VisionMambaDual(ssm_fn, img_size=img_size, embed_dim=128, depth=4)
