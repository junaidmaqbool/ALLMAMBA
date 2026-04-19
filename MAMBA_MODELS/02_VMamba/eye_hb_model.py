"""
eye_hb_model.py  --  02_VMamba
Loads the simplified single-file VMamba (VSSM) from vmamba.py.

VSSM(depths, dims, num_classes=0) returns (B, C) features.
We wrap it with make_dual_head() for (logits[B,2], hb_pred[B,1]).

Falls back to ResNet-18 if vmamba.py fails to load.
"""

import os, sys, torch, torch.nn as nn

_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.dirname(_THIS_DIR)
for p in [_MODELS_DIR, _THIS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from common.pure_ssm import make_dual_head, _PureMamba, _VisionMambaDual


def build_model(img_size: int = 224):
    """
    Returns VMamba model with forward(x) -> (logits[B,2], hb_pred[B,1]).
    """
    vmamba_py = os.path.join(_THIS_DIR, "vmamba.py")

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("_vmamba_mod", vmamba_py)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Tiny VMamba: 4 stages, dims 96->192->384->768
        backbone = mod.VSSM(
            depths=[2, 2, 4, 2],
            dims=[96, 192, 384, 768],
            num_classes=0,
            imgsize=img_size,
        )
        feat_dim = backbone.num_features  # 768
        print(f"  [VMamba] VSSM loaded from vmamba.py  (feat_dim={feat_dim})")

        # VSSM.forward() with num_classes=0 passes through
        # norm+Identity head -> returns (B, feat_dim) tensor.
        # We wrap it so our dual head replaces that final step.
        class _VMambaWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.vssm      = backbone
                self.pool      = nn.AdaptiveAvgPool2d(1)
                self.cls_head  = nn.Linear(feat_dim, 2)
                self.reg_head  = nn.Sequential(
                    nn.Linear(feat_dim, feat_dim // 4), nn.GELU(),
                    nn.Linear(feat_dim // 4, 1))

            def _extract_feat(self, x):
                """Run VSSM and return flat (B, C) feature vector."""
                x = self.vssm.patch_embed(x)
                if getattr(self.vssm, 'pos_embed', None) is not None:
                    pos = self.vssm.pos_embed
                    if pos.dim() == 4:
                        pos = pos.permute(0, 2, 3, 1) if not self.vssm.channel_first else pos
                    x = x + pos
                for layer in self.vssm.layers:
                    x = layer(x)
                # x: (B, H, W, C) channel-last  OR  (B, C, H, W) channel-first
                cf = getattr(self.vssm, 'channel_first', False)
                if cf:
                    x = self.pool(x).flatten(1)          # (B,C,H,W) -> (B,C)
                else:
                    x = x.mean([1, 2])                   # (B,H,W,C) -> (B,C)
                return x

            def forward(self, x):
                f = self._extract_feat(x)
                return self.cls_head(f), self.reg_head(f)

        return _VMambaWrapper()

    except Exception as e:
        print(f"  [VMamba] vmamba.py load failed ({e})  ->  ResNet-18 fallback.")
        from torchvision.models import resnet18
        bb = resnet18(weights=None)
        bb.fc = nn.Identity()
        return make_dual_head(bb, feat_dim=512)
