"""Known-face DB: walks KNOWN_DIR/*.npy → loads mean L2-normalized embed per person.

Mutates `app.core.state.KNOWN_EMBS` and `KNOWN_NAMES` so the rest of the
pipeline can read them through the central state module.
"""

import os
import numpy as np
import torch

from app.core.config import DEVICE, KNOWN_DIR
from app.core import state as app_state


def reload_database():
    """Walk KNOWN_DIR/*.npy. Each file = one person; name = filename stem.
    File may store (1, 512) or (K, 512) — take mean, L2-normalize, stack."""
    os.makedirs(KNOWN_DIR, exist_ok=True)
    rows, names = [], []
    for fname in sorted(os.listdir(KNOWN_DIR)):
        if not fname.endswith('.npy'):
            continue
        path = os.path.join(KNOWN_DIR, fname)
        try:
            arr = np.load(path)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            mean = arr.mean(axis=0)
            mean = mean / (np.linalg.norm(mean) + 1e-8)
            rows.append(mean.astype(np.float32))
            names.append(os.path.splitext(fname)[0])
        except Exception as e:
            print(f"[CACHE ERROR] {fname}: {e}")
    if rows:
        app_state.KNOWN_EMBS = torch.from_numpy(np.stack(rows, axis=0)).to(DEVICE).float()
        app_state.KNOWN_NAMES = names
        print(f"[CACHE] Database loaded: {len(app_state.KNOWN_NAMES)} people from {KNOWN_DIR}")
    else:
        app_state.KNOWN_EMBS = None
        app_state.KNOWN_NAMES = []


def save_to_db(name, vector):
    """Write known/<name>.npy as (1, 512). Overwrites if exists."""
    os.makedirs(KNOWN_DIR, exist_ok=True)
    arr = vector.cpu().numpy() if torch.is_tensor(vector) else np.asarray(vector)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    safe = name.replace('/', '_').replace('\\', '_')
    np.save(os.path.join(KNOWN_DIR, f"{safe}.npy"), arr.astype(np.float32))
    reload_database()
    print(f"[DB] Registered: {safe}.npy")
