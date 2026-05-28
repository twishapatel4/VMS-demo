"""OSNet x0_25 body ReID loader.

Architecture lives in osnet_arch.py (vendored). Loader strips the classifier
head — we only use eval-mode features. Returns None on missing checkpoint so
the rest of the pipeline can disable body ReID gracefully.
"""

import os
import torch

from app.core.config import DEVICE, OSNET_CHECKPOINT
from .osnet_arch import osnet_x0_25

OSNET_INPUT_H, OSNET_INPUT_W = 256, 128
OSNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
OSNET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)


def load_osnet():
    if not os.path.exists(OSNET_CHECKPOINT):
        print(f"[WARN] OSNet checkpoint missing at {OSNET_CHECKPOINT}. Body ReID disabled.")
        return None
    try:
        model = osnet_x0_25(num_classes=1000, pretrained=False, loss='softmax')
        ckpt = torch.load(OSNET_CHECKPOINT, map_location=DEVICE)
        state = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
        # Strip classifier head — we only need features in eval mode.
        state = {k: v for k, v in state.items() if not k.startswith('classifier')}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            print(f"[WARN] OSNet unexpected keys: {len(unexpected)} (ignored)")
        model.eval().to(DEVICE)
        print("[SYSTEM] OSNet loaded (body ReID active).")
        return model
    except Exception as e:
        print(f"[WARN] OSNet load failed: {e}. Body ReID disabled.")
        return None
