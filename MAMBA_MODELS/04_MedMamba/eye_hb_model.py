"""
eye_hb_model.py  --  04_MedMamba
Loads MedMamba VSSM from MedMamba.py for dual-task HB prediction.

MedMamba.py has module-level `.to("cuda")` calls that crash on import
when no GPU is available.  We neutralise them with a temporary no-op.

VSSM(depths=[2,2,4,2], dims=[96,192,384,768], num_classes=0):
    forward_backbone(x) -> (B, H, W, C) channel-last
    forward(x) -> (B, 768)  [avgpool + flatten + Identity head]

Falls back to ConvNeXt-Tiny if import fails.
"""

import os, sys, types, torch, torch.nn as nn

_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.dirname(_THIS_DIR)
for p in [_MODELS_DIR, _THIS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from common.pure_ssm import make_dual_head


def build_model(img_size: int = 224):
    """
    Returns MedMamba model with forward(x) -> (logits[B,2], hb_pred[B,1]).
    """
    medmamba_py = os.path.join(_THIS_DIR, "MedMamba.py")

    try:
        import importlib.util

        # ── Disable .to() at module level (MedMamba.py calls .to("cuda") globally)
        _orig_to = nn.Module.to
        nn.Module.to = lambda self, *a, **kw: self   # no-op

        spec    = importlib.util.spec_from_file_location("_medmamba", medmamba_py)
        med_mod = types.ModuleType("_medmamba")
        try:
            spec.loader.exec_module(med_mod)
        finally:
            nn.Module.to = _orig_to   # always restore regardless of errors

        VSSM = med_mod.VSSM
        backbone = VSSM(
            depths=[2, 2, 4, 2],
            dims=[96, 192, 384, 768],
            num_classes=0,
        )
        feat_dim = backbone.num_features   # 768

        class _MedWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = backbone
                self.pool     = nn.AdaptiveAvgPool2d(1)
                self.cls_head = nn.Linear(feat_dim, 2)
                self.reg_head = nn.Sequential(
                    nn.Linear(feat_dim, feat_dim // 4), nn.GELU(),
                    nn.Linear(feat_dim // 4, 1))

            def forward(self, x):
                # forward() with num_classes=0 returns (B, C) already
                f = self.backbone(x)            # (B, feat_dim)
                if f.dim() == 4:               # safety: (B,H,W,C) or (B,C,H,W)
                    if f.shape[1] == feat_dim:
                        f = self.pool(f).flatten(1)
                    else:
                        f = f.mean([1, 2])
                return self.cls_head(f), self.reg_head(f)

        print(f"  [MedMamba] VSSM loaded from MedMamba.py  (feat_dim={feat_dim})")
        return _MedWrapper()

    except Exception as e:
        print(f"  [MedMamba] load failed ({e})  ->  ConvNeXt-Tiny fallback.")
        from torchvision.models import convnext_tiny
        bb = convnext_tiny(weights=None)
        bb.classifier[-1] = nn.Identity()
        return make_dual_head(bb, feat_dim=768)
