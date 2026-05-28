"""AdaFace IR-101 face embedder loader.

Architecture lives in adaface_arch.py (vendored from AdaFace repo). This module
loads the checkpoint, normalizes the state-dict keys, and returns a frozen,
eval-mode model on DEVICE.
"""

import torch

from app.core.config import DEVICE
from .adaface_arch import build_model


def load_adaface(path):
    model = build_model('ir_101')
    try:
        checkpoint = torch.load(path, map_location=DEVICE)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        model_dict = model.state_dict()
        new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()
                          if k.replace('model.', '') in model_dict}
        model.load_state_dict(new_state_dict, strict=False)
        model.to(DEVICE)
        # model.half()  # Use half precision for faster inference on compatible GPUs
        model.eval()
        return model
    except Exception as e:
        print(f"[FATAL] AdaFace Load Error: {e}")
        exit()
