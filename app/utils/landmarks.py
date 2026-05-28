"""Face landmark utilities: ArcFace alignment + sanity checks + frontality.

Used by both the detection filter (Stage 1/Stage 2 gates) and the embedding
prep step (norm_crop aligns to ArcFace 5-point template before AdaFace forward).
"""

import cv2
import numpy as np

from app.core.config import FRONTAL_REF, FRONTAL_WEIGHT_FLOOR

# ArcFace 5-point template for 112x112 alignment.
ARCFACE_DST = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014],
    [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]
], dtype=np.float32)


def norm_crop(img, landmarks, size=112):
    M, _ = cv2.estimateAffinePartial2D(landmarks, ARCFACE_DST, method=cv2.LMEDS)
    if M is None:
        return np.zeros((size, size, 3), dtype=np.uint8)
    return cv2.warpAffine(img, M, (size, size), borderValue=0.0)


def valid_landmarks(lmks, box):
    """LOOSE landmark sanity — used by Stage 1 (ByteTrack input).

    Pre-FLOW2 checks only. Permissive enough that profile / hand-on-face
    detections still feed the tracker, keeping tracks alive during occlusion.
    Stricter geometry lives in valid_landmarks_strict() below.
    """
    x1, y1, x2, y2 = box
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    margin = 0.15 * max(w, h)   # widened from 0.05 — profile far-side landmarks often near/past bbox edge

    # All landmarks must sit (mostly) inside the bbox.
    if (lmks[:, 0].min() < x1 - margin or lmks[:, 0].max() > x2 + margin or
        lmks[:, 1].min() < y1 - margin or lmks[:, 1].max() > y2 + margin):
        return False

    # Only reject if eyes collapse to a single point (true garbage). Profile eye_dist ~0.05*w is fine.
    eye_dist = float(np.linalg.norm(lmks[0] - lmks[1]))
    if eye_dist < 0.02 * w:
        return False

    # Eyes must be above mouth corners (image y increases downward).
    if lmks[0, 1] > lmks[3, 1] or lmks[1, 1] > lmks[4, 1]:
        return False

    return True


def valid_landmarks_strict(lmks, box):
    """STRICT geometry — used by Stage 2 (identify_ok flag for AdaFace embed).

    Rejects back-of-head / hair-patch fakes where RetinaFace fabricates 5 landmarks
    that pass loose checks but violate real-face geometry. Tracking is unaffected:
    failing this only suppresses identification on the current frame.
    See [FLOW2-RISK #8] for revert.
    """
    x1, y1, x2, y2 = box
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    re, le = lmks[0], lmks[1]
    nose = lmks[2]
    rm, lm = lmks[3], lmks[4]

    eye_dist = float(np.linalg.norm(re - le))
    if eye_dist < 1e-3:
        return False

    # Nose horizontal sanity. Real face: nose between eyes (allow ±30% of eye span for profile yaw).
    eye_x_min, eye_x_max = min(re[0], le[0]), max(re[0], le[0])
    eye_span = max(eye_x_max - eye_x_min, 1.0)
    margin_x = 0.3 * eye_span
    if nose[0] < eye_x_min - margin_x or nose[0] > eye_x_max + margin_x:
        return False

    # Nose vertical sanity: between eye line and mouth line (with small tolerance).
    eye_y = 0.5 * (re[1] + le[1])
    mouth_y = 0.5 * (rm[1] + lm[1])
    if nose[1] < eye_y - 0.10 * h or nose[1] > mouth_y + 0.10 * h:
        return False

    # Mouth/eye width ratio sanity. Real face ratio ~0.5-1.5; back-of-head fakes random.
    mouth_dist = float(np.linalg.norm(rm - lm))
    ratio = mouth_dist / eye_dist
    if ratio < 0.4 or ratio > 1.8:
        return False

    return True


def frontality_score(lmks):
    """Cheap yaw proxy from 5 landmarks. Returns [0, 1]: 1.0 = frontal, 0.0 = full profile.
    Cues: (a) eye-separation ratio (profile collapses eyes in 2D),
          (b) nose horizontal offset from eye midpoint (profile shifts nose far)."""
    re, le = lmks[0], lmks[1]
    nose = lmks[2]
    rm, lm = lmks[3], lmks[4]

    eye_dist   = float(np.linalg.norm(re - le))
    mouth_dist = float(np.linalg.norm(rm - lm))
    face_w = max(eye_dist, mouth_dist) * 2.5
    if face_w < 1e-3:
        return 0.0

    eye_ratio  = eye_dist / face_w
    eye_mid_x  = 0.5 * (re[0] + le[0])
    nose_off   = abs(nose[0] - eye_mid_x) / max(eye_dist, 1e-3)

    eye_score  = min(1.0, eye_ratio / 0.30)
    nose_score = max(0.0, 1.0 - nose_off / 0.50)
    return float(0.5 * eye_score + 0.5 * nose_score)


def frontality_weight(fscore):
    """Map frontality [0,1] → embed contribution weight, clipped to [FLOOR, 1.0]."""
    w = fscore / max(FRONTAL_REF, 1e-6)
    if w > 1.0:
        w = 1.0
    if w < FRONTAL_WEIGHT_FLOOR:
        w = FRONTAL_WEIGHT_FLOOR
    return float(w)
