"""RetinaFace detector loader.

Architecture + utils live in app/vendor/retinaface/. This module just wraps
checkpoint loading + device placement.
"""

import torch

from app.core.config import DEVICE
from app.vendor.retinaface.retinaface import RetinaFace


def load_retinaface(path, network='mobile0.25', half=False):
    model = RetinaFace(network_name=network, half=half, device=DEVICE)
    try:
        state = torch.load(path, map_location=lambda s, l: s)
        # strip DataParallel 'module.' prefix if present
        state = {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        model.eval()
        model = model.to(DEVICE)
        return model
    except Exception as e:
        print(f"[FATAL] RetinaFace Load Error: {e}")
        exit()
