"""Track-stitching helper [FLOW2-RISK #11].

When ByteTrack reassigns a fresh tid for the same person (e.g. after >6s of
no usable detection), this helper recovers the previous label without running
a 3-frame vote — by checking bbox overlap against recently-lost labels in
the cam's stitch_cache.
"""

from app.core.config import STITCH_WINDOW_FRAMES, STITCH_IOU_MIN
from app.utils.geometry import iou_xyxy


def try_stitch_label(cam_state, track_box, frame_idx, alive_tids):
    """Returns dict {'label','color','eid'} on hit or None on miss. Also prunes
    stitch_cache entries older than STITCH_WINDOW_FRAMES.

    Skips cache entries belonging to currently-alive labeled tids — those are
    the existing person, not a candidate for stitching onto a different tid.
    """
    if not cam_state.stitch_cache:
        return None

    # Compute set of (eid, name) currently alive so we don't steal someone's label.
    alive_eids = {cam_state.track_to_eid[t] for t in alive_tids if t in cam_state.track_to_eid}
    alive_known_names = {cam_state.track_to_label[t][0] for t in alive_tids
                         if t in cam_state.track_to_label and t not in cam_state.track_to_eid}

    expired = []
    best_match, best_iou = None, 0.0
    for key, info in cam_state.stitch_cache.items():
        age = frame_idx - info['frame_idx']
        if age > STITCH_WINDOW_FRAMES:
            expired.append(key)
            continue
        # Don't stitch to a label that's already drawn on another live tid.
        if key[0] == 'eid' and key[1] in alive_eids:
            continue
        if key[0] == 'known' and key[1] in alive_known_names:
            continue
        iou = iou_xyxy(track_box, info['box'])
        if iou >= STITCH_IOU_MIN and iou > best_iou:
            best_iou = iou
            best_match = info

    for k in expired:
        cam_state.stitch_cache.pop(k, None)

    return best_match
