"""Body-crop estimation for OSNet ReID. Face box → expanded body region."""

from app.core.config import BODY_CROP_MIN_W, BODY_CROP_MIN_H


def body_crop(frame, box, scale_h=3.5, scale_w=1.5):
    """Expand face box downward to estimate full-body region.

    Returns None if the resulting crop is too small to be useful for OSNet —
    this is what FLOW2's 'body crop valid?' gate checks.
    See [FLOW2-RISK #7] BODY_CROP_MIN_W / BODY_CROP_MIN_H in config.
    """
    x1, y1, x2, y2 = box
    fh, fw = y2 - y1, x2 - x1
    cx = (x1 + x2) // 2
    new_w = int(fw * scale_w)
    bx1 = max(0, cx - new_w // 2)
    bx2 = min(frame.shape[1], cx + new_w // 2)
    by1 = max(0, y1 - int(fh * 0.2))
    by2 = min(frame.shape[0], y1 + int(fh * scale_h))
    crop = frame[by1:by2, bx1:bx2]
    if crop.size == 0:
        return None
    h, w = crop.shape[:2]
    # [FLOW2-RISK #7] body-crop validity gate. See constant defs.
    if w < BODY_CROP_MIN_W or h < BODY_CROP_MIN_H:
        return None
    return crop
