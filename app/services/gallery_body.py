"""Body-ReID gallery (FLOW2 hybrid).

Same UnknownGallery class as the face side, but constructed with the OSNet
match threshold. Files live in BODY_DIR keyed by the SAME eid as the face
file. Body file may be absent for an eid that was created via the body-invalid
LEFT branch (face-only entry).
"""

import os
import time
from collections import deque

import numpy as np
import torch

from app.core.config import (
    DEVICE,
    BODY_DIR,
    OSNET_THRESH,
    UNKNOWN_TTL_SECONDS,
    TOPK_GALLERY,
)
from .gallery_unknown import UnknownGallery


body_gallery = UnknownGallery(
    match_thresh=OSNET_THRESH,
    ttl=UNKNOWN_TTL_SECONDS,
    maxlen=TOPK_GALLERY,
)


def _body_path(eid):
    return os.path.join(BODY_DIR, f"Body_{eid:03d}.npy")


def persist_body(eid):
    with body_gallery.lock:
        e = body_gallery.entries.get(eid)
        if e is None or not e['embeds']:
            return
        stack = torch.cat(list(e['embeds']), dim=0).cpu().numpy().astype(np.float32)
    os.makedirs(BODY_DIR, exist_ok=True)
    np.save(_body_path(eid), stack)


def delete_body_file(eid):
    p = _body_path(eid)
    if os.path.isfile(p):
        try:
            os.remove(p)
        except OSError:
            pass


def wipe_body_dir():
    if not os.path.isdir(BODY_DIR):
        return
    for fname in os.listdir(BODY_DIR):
        if fname.startswith('Body_') and fname.endswith('.npy'):
            try:
                os.remove(os.path.join(BODY_DIR, fname))
            except OSError:
                pass


def load_bodies_from_disk():
    """Repopulate body_gallery from BODY_DIR/*.npy."""
    if not os.path.isdir(BODY_DIR):
        return
    now = time.time()
    loaded = 0
    with body_gallery.lock:
        for fname in sorted(os.listdir(BODY_DIR)):
            if not (fname.startswith('Body_') and fname.endswith('.npy')):
                continue
            try:
                eid = int(fname[len('Body_'):-len('.npy')])
            except ValueError:
                continue
            try:
                arr = np.load(os.path.join(BODY_DIR, fname))
            except Exception:
                continue
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            embeds = deque(maxlen=body_gallery.maxlen)
            for row in arr:
                t = torch.from_numpy(row.reshape(1, -1)).to(DEVICE).float()
                t = t / (torch.norm(t, dim=1, keepdim=True) + 1e-8)
                embeds.append(t)
            body_gallery.entries[eid] = {
                'embeds': embeds,
                'last_seen': now,
                'count': arr.shape[0],
            }
            body_gallery.next_id = max(body_gallery.next_id, eid + 1)
            loaded += 1
    if loaded:
        print(f"[CACHE] Loaded {loaded} body entries from {BODY_DIR}")


def persist_all_bodies():
    os.makedirs(BODY_DIR, exist_ok=True)
    with body_gallery.lock:
        for eid, e in body_gallery.entries.items():
            if not e['embeds']:
                continue
            stack = torch.cat(list(e['embeds']), dim=0).cpu().numpy().astype(np.float32)
            np.save(_body_path(eid), stack)
