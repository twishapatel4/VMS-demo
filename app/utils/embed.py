"""Embedding forward-pass helpers — AdaFace face embed + OSNet body embed.

Both return L2-normalized fp32 tensors on DEVICE.

Models are injected at call-time (not imported at module scope) so this file
stays free of side-effects and can be imported during config-loading without
triggering model loads.
"""

import cv2
import numpy as np
import torch

from app.core.config import DEVICE
from app.models.osnet import OSNET_INPUT_H, OSNET_INPUT_W, OSNET_MEAN, OSNET_STD


def embed_batch(adaface, aligned_batch):
    """Run AdaFace forward on a [N, 3, 112, 112] uint8/float RGB batch in [0,255].
    Returns [N, 512] L2-normalized fp32 tensor on DEVICE."""
    x = (aligned_batch - 127.5) / 128.0
    x = x.to(DEVICE)  # Use half precision for faster inference on compatible GPUs
    # .half()
    with torch.no_grad():
        feats, _ = adaface(x)
    feats = feats.float()  # cast back to fp32 for DB matmul (KNOWN_EMBS is fp32)
    feats = feats / (torch.norm(feats, dim=1, keepdim=True) + 1e-8)
    return feats


def osnet_embed(osnet, crops_bgr):
    """Extract OSNet body embeddings. crops_bgr: list of BGR numpy arrays (variable size).
    Returns [N, feature_dim] L2-normalized fp32 tensor on DEVICE."""
    batch = []
    for crop in crops_bgr:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (OSNET_INPUT_W, OSNET_INPUT_H))   # cv2 expects (W, H)
        batch.append(rgb)
    x = np.stack(batch, axis=0).astype(np.float32) / 255.0
    x = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous().to(DEVICE)
    x = (x - OSNET_MEAN) / OSNET_STD
    with torch.no_grad():
        feats = osnet(x)
    feats = feats.float()
    feats = feats / (torch.norm(feats, dim=1, keepdim=True) + 1e-8)
    return feats
