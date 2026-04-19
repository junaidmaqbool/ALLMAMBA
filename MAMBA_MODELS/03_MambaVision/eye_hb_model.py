"""
eye_hb_model.py  --  03_MambaVision
Loads MambaVision-T from the local mambavision package.

Architecture: Hybrid Mamba + Transformer (NVIDIA, CVPR 2025).
MambaVision.forward_features(x) -> (B, 640) flat vector.
We attach dual heads for (logits[B,2], hb_pred[B,1]).

Falls back to EfficientNet-B0 if mambavision fails to load.
"""

import os, sys, torch, torch.nn as nn

_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.dirname(_THIS_DIR)
for p in [_MODELS_DIR, _THIS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from common.pure_ssm import make_dual_head


def build_model(img_size: int = 224):
    """
    Returns MambaVision-T with forward(x) -> (logits[B,2], hb_pred[B,1]).
    """
    mv_models_dir = os.path.join(_THIS_DIR, "mambavision", "models")
    if os.path.join(_THIS_DIR, "mambavision") not in sys.path:
        sys.path.insert(0, os.path.join(_THIS_DIR, "mambavision"))
    if mv_models_dir not in sys.path:
        sys.path.insert(0, mv_models_dir)

    try:
        # MambaVision-T params:
        #   dim=80, depths=[1,3,8,4], num_heads=[2,4,8,16]
        #   window_size=[8,8,14,7]
        #   num_features = 80 * 2^3 = 640
        from mamba_vision import MambaVision

        backbone = MambaVision(
            dim=80,
            in_dim=32,
            depths=[1, 3, 8, 4],
            window_size=[8, 8, 14, 7],
            mlp_ratio=4,
            num_heads=[2, 4, 8, 16],
            resolution=img_size,
            drop_path_rate=0.2,
            num_classes=0,      # head = Identity, forward returns flat features
        )
        feat_dim = 80 * (2 ** 3)   # 640

        class _MVWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone  = backbone
                self.cls_head  = nn.Linear(feat_dim, 2)
                self.reg_head  = nn.Sequential(
                    nn.Linear(feat_dim, feat_dim // 4), nn.GELU(),
                    nn.Linear(feat_dim // 4, 1))

            def forward(self, x):
                # forward_features returns (B, feat_dim) after avgpool+flatten
                f = self.backbone.forward_features(x)
                return self.cls_head(f), self.reg_head(f)

        print(f"  [MambaVision] MambaVision-T loaded from local mambavision/  (feat_dim={feat_dim})")
        return _MVWrapper()

    except Exception as e:
        print(f"  [MambaVision] local load failed ({e})  ->  EfficientNet-B0 fallback.")
        from torchvision.models import efficientnet_b0
        bb      = efficientnet_b0(weights=None)
        bb.classifier[-1] = nn.Identity()
        return make_dual_head(bb, feat_dim=1280)
