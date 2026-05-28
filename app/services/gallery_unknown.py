"""Unknown-face short-term gallery (RAM + disk persistence).

UnknownGallery is a thread-safe top-K embed store keyed by an auto-incrementing
eid. Same person seen across frames → same Unknown_XXX eid via cosine match.

This module instantiates the shared `unknown_gallery` singleton used by the
face cascade. The body gallery (same class, different threshold) is created
in gallery_body.py.

Also owns: disk persistence helpers + invalidate_eid_bindings (which touches
per-cam track state across all cameras).
"""

import os
import threading
import time
from collections import deque

import numpy as np
import torch

from app.core.config import (
    DEVICE,
    UNKNOWN_DIR,
    UNKNOWN_THRESH,
    UNKNOWN_TTL_SECONDS,
    UNKNOWN_EMA_ALPHA,
    TOPK_GALLERY,
)
from app.core import state as app_state


class UnknownGallery:
    """RAM-only short-term memory of strangers. Same person → same Unknown_XXX ID across frames.

    Top-K design: each entry stores a deque of up to `maxlen` embeddings (FIFO).
    Matching computes cosine sim against ALL stored embeddings per identity, takes max.
    Better cross-angle recall than a single EMA centroid (which gets muddy after
    averaging frontal + side faces)."""

    def __init__(self, match_thresh=UNKNOWN_THRESH, ttl=UNKNOWN_TTL_SECONDS, maxlen=TOPK_GALLERY):
        # id -> {'embeds': deque[tensor[1,512]], 'last_seen': epoch, 'count': int}
        self.entries = {}
        self.next_id = 1
        self.match_thresh = match_thresh
        self.ttl = ttl
        self.maxlen = maxlen
        self.lock = threading.Lock()

    def _best_match_unsafe(self, feat):
        """Caller must hold lock. Returns (best_eid, best_sim) scanning all stored embeds."""
        if not self.entries:
            return None, 0.0
        best_eid, best_sim = None, -1.0
        for eid, e in self.entries.items():
            if not e['embeds']:
                continue
            stack = torch.cat(list(e['embeds']), dim=0)        # [K_i, 512]
            sims = torch.mm(feat, stack.t())                   # [1, K_i]
            s = float(sims.max().item())
            if s > best_sim:
                best_sim = s
                best_eid = eid
        return best_eid, best_sim

    def assign(self, feat, now, weight=1.0):
        """Returns (eid, sim). Reuses entry (EMA-blend slot) if match, else creates new."""
        with self.lock:
            best_eid, best_sim = self._best_match_unsafe(feat)
            if best_eid is not None and best_sim >= self.match_thresh:
                self._ema_blend_unsafe(best_eid, feat, now, weight=weight)
                return best_eid, best_sim
            eid = self.next_id
            self.next_id += 1
            self.entries[eid] = {
                'embeds': deque([feat.detach().clone()], maxlen=self.maxlen),
                'last_seen': now,
                'count': 1,
            }
            return eid, 0.0

    def _ema_blend_unsafe(self, eid, feat, now, weight=1.0):
        """Caller must hold lock. K=1 EMA: blend fresh embed into single stored slot.
        `weight` in [0,1] scales effective alpha — frontal frame uses full α, profile uses fraction."""
        e = self.entries[eid]
        if e['embeds']:
            alpha = UNKNOWN_EMA_ALPHA * float(weight)
            old = e['embeds'][0]
            blended = alpha * feat + (1.0 - alpha) * old
            blended = blended / (torch.norm(blended, dim=1, keepdim=True) + 1e-8)
            e['embeds'][0] = blended.detach().clone()
        else:
            e['embeds'].append(feat.detach().clone())
        e['last_seen'] = now
        e['count'] += 1

    def find_match(self, feat):
        """Read-only lookup. Returns (eid, sim) if above threshold, else (None, best_sim)."""
        with self.lock:
            best_eid, best_sim = self._best_match_unsafe(feat)
            if best_eid is not None and best_sim >= self.match_thresh:
                return best_eid, best_sim
            return None, best_sim

    def update_entry(self, eid, feat, now, weight=1.0):
        """EMA-blend fresh embed into stored slot, scaled by weight.

        If eid is missing, create the entry at that eid. Required so body_gallery
        (which always follows unknown_gallery's auto-incremented eid) can be
        populated without auto-incrementing its own next_id.
        """
        with self.lock:
            if eid in self.entries:
                self._ema_blend_unsafe(eid, feat, now, weight=weight)
            else:
                # [BUG-FIX] previously this was a no-op when eid missing; body_gallery
                # therefore never accumulated entries. Create-on-miss so FLOW2 hybrid
                # cascade can attach body embeds at the eid chosen by unknown_gallery.
                self.entries[eid] = {
                    'embeds': deque([feat.detach().clone()], maxlen=self.maxlen),
                    'last_seen': now,
                    'count': 1,
                }
                self.next_id = max(self.next_id, eid + 1)

    def evict(self, now):
        with self.lock:
            stale = [k for k, v in self.entries.items() if now - v['last_seen'] > self.ttl]
            for k in stale:
                del self.entries[k]

    def get_feat(self, eid):
        """Return L2-normalized mean of stored embeds (used for DB registration)."""
        with self.lock:
            e = self.entries.get(eid)
            if e is None or not e['embeds']:
                return None
            stack = torch.cat(list(e['embeds']), dim=0)
            mean = stack.mean(dim=0, keepdim=True)
            mean = mean / (torch.norm(mean, dim=1, keepdim=True) + 1e-8)
            return mean

    def remove(self, eid):
        with self.lock:
            self.entries.pop(eid, None)

    def clear(self):
        with self.lock:
            count = len(self.entries)
            self.entries.clear()
            self.next_id = 1
            return count

    def snapshot(self):
        with self.lock:
            return {k: {'count': v['count'], 'last_seen': v['last_seen'], 'k': len(v['embeds'])}
                    for k, v in self.entries.items()}


# Shared singleton — face-side unknown gallery.
unknown_gallery = UnknownGallery()


# --- UNKNOWN GALLERY DISK PERSISTENCE ---
def _unknown_path(eid):
    return os.path.join(UNKNOWN_DIR, f"Unknown_{eid:03d}.npy")


def persist_unknown(eid):
    """Snapshot one gallery entry to disk. Overwrites file."""
    with unknown_gallery.lock:
        e = unknown_gallery.entries.get(eid)
        if e is None or not e['embeds']:
            return
        stack = torch.cat(list(e['embeds']), dim=0).cpu().numpy().astype(np.float32)
    os.makedirs(UNKNOWN_DIR, exist_ok=True)
    np.save(_unknown_path(eid), stack)


def delete_unknown_file(eid):
    p = _unknown_path(eid)
    if os.path.isfile(p):
        try:
            os.remove(p)
        except OSError:
            pass


def wipe_unknown_dir():
    if not os.path.isdir(UNKNOWN_DIR):
        return
    for fname in os.listdir(UNKNOWN_DIR):
        if fname.startswith('Unknown_') and fname.endswith('.npy'):
            try:
                os.remove(os.path.join(UNKNOWN_DIR, fname))
            except OSError:
                pass


def load_unknowns_from_disk():
    """Repopulate unknown_gallery from UNKNOWN_DIR/*.npy. Sets next_id beyond max loaded eid."""
    if not os.path.isdir(UNKNOWN_DIR):
        return
    now = time.time()
    loaded = 0
    with unknown_gallery.lock:
        for fname in sorted(os.listdir(UNKNOWN_DIR)):
            if not (fname.startswith('Unknown_') and fname.endswith('.npy')):
                continue
            try:
                eid = int(fname[len('Unknown_'):-len('.npy')])
            except ValueError:
                continue
            try:
                arr = np.load(os.path.join(UNKNOWN_DIR, fname))
            except Exception:
                continue
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            embeds = deque(maxlen=unknown_gallery.maxlen)
            for row in arr:
                t = torch.from_numpy(row.reshape(1, -1)).to(DEVICE).float()
                t = t / (torch.norm(t, dim=1, keepdim=True) + 1e-8)
                embeds.append(t)
            unknown_gallery.entries[eid] = {
                'embeds': embeds,
                'last_seen': now,
                'count': arr.shape[0],
            }
            unknown_gallery.next_id = max(unknown_gallery.next_id, eid + 1)
            loaded += 1
    if loaded:
        print(f"[CACHE] Loaded {loaded} unknown entries from {UNKNOWN_DIR}")


def persist_all_unknowns():
    """Bulk save all current gallery entries (used at exit)."""
    os.makedirs(UNKNOWN_DIR, exist_ok=True)
    with unknown_gallery.lock:
        for eid, e in unknown_gallery.entries.items():
            if not e['embeds']:
                continue
            stack = torch.cat(list(e['embeds']), dim=0).cpu().numpy().astype(np.float32)
            np.save(_unknown_path(eid), stack)


def invalidate_eid_bindings(eid):
    """Drop any track-cache binding pointing to a removed/promoted eid across all cams.
    Without this, Branch A keeps drawing the cached (red Unknown_NNN) label until ByteTrack
    kills the tid — even though the eid no longer exists in the unknown gallery."""
    cams = app_state.cam_states
    if not cams:
        return
    for cs in cams.values():
        dead = [t for t, e in cs.track_to_eid.items() if e == eid]
        for t in dead:
            cs.reset_track(t)
        # [FLOW2-RISK #11] also drop stitch cache for the promoted eid so the
        # old red "Unknown_NNN" doesn't get re-stitched onto a fresh tid.
        cs.drop_stitch_for_eid(eid)
