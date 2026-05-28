"""Mutable runtime state shared across services.

Holds:
  - Known-face DB cache (KNOWN_EMBS, KNOWN_NAMES) — refreshed by gallery_known.
  - CamState class — per-camera ByteTrack + tid bindings + stitch cache.
  - Process-level cam_states map + admin naming flag.

These globals exist because the legacy main.py read them everywhere; keeping
them centralized makes the dependency explicit instead of hiding it across
imports.
"""

from .config import (
    BYTETRACK_TRACK_THRESH,
    BYTETRACK_MATCH_THRESH,
    BYTETRACK_BUFFER,
    BYTETRACK_FRAME_RATE,
)
from app.vendor.bytetrack import BYTETracker

# --- KNOWN-FACE DB CACHE (populated by services.gallery_known.reload_database) ---
KNOWN_EMBS = None
KNOWN_NAMES = []

# --- ADMIN NAMING FLAG ---
is_naming = False

# --- PER-CAM STATE MAP (populated by main at startup) ---
cam_states = {}


class CamState:
    """Per-camera tracking state. ByteTracker + tid bindings + per-track embed buffers."""

    def __init__(self, label):
        self.label = label
        self.tracker = BYTETracker(
            track_thresh=BYTETRACK_TRACK_THRESH,
            match_thresh=BYTETRACK_MATCH_THRESH,
            track_buffer=BYTETRACK_BUFFER,
            frame_rate=BYTETRACK_FRAME_RATE,
        )
        self.track_to_label   = {}   # tid -> (name, color)
        self.track_to_eid     = {}   # tid -> eid (only set for unknown bindings; known DB hits omitted)
        self.track_query_buf  = {}   # tid -> list[tensor[1,512]] embeds accumulating for vote
        self.track_last_embed = {}   # tid -> frame_idx of last gallery-update embed
        # [FLOW2-RISK #11] stitch_cache: cache_key -> {'label','color','eid','box','frame_idx'}
        # Updated every frame from labeled tids (Branch A); read by Branch B on new tid
        # to inherit label without 3-frame vote when bbox overlaps recently-lost label.
        self.stitch_cache     = {}

    def reset_track(self, tid):
        """Drop a tid from all per-track maps. Next detect frame re-votes via cascade.
        stitch_cache intentionally NOT touched here — we want recently-lost labels
        to remain available for spatial-rebind via Branch B stitching."""
        self.track_to_label.pop(tid, None)
        self.track_to_eid.pop(tid, None)
        self.track_query_buf.pop(tid, None)
        self.track_last_embed.pop(tid, None)

    def reset_all_tracks(self):
        """Drop every binding. Used by 'clear' admin op — also wipes stitch_cache."""
        self.track_to_label.clear()
        self.track_to_eid.clear()
        self.track_query_buf.clear()
        self.track_last_embed.clear()
        self.stitch_cache.clear()

    def drop_stitch_for_eid(self, eid):
        """Remove stitch_cache entry for a given eid (called after register/promotion
        so the old Unknown_NNN label isn't re-stitched onto a new track of the
        now-promoted person)."""
        self.stitch_cache.pop(('eid', eid), None)
