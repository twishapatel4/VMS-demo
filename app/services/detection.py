"""Detection stage: RetinaFace forward + two-stage filter.

Stage 1 (size + aspect + LOOSE landmarks) → ByteTracker input.
Stage 2 (STRICT landmarks + frontality) → identify_ok flag for AdaFace embed.

[FLOW2-RISK #9] two-stage detection filter. Stage 1 keeps tracks alive during
occlusion / profile turns / hand-on-face. Stage 2 prevents bad embeds from
polluting the gallery on those same frames.
"""

import cv2
import numpy as np
import torch

from app.core.config import (
    DET_THRESH,
    NMS_THRESH,
    DETECT_SCALE,
    MIN_FACE_SIZE,
    ASPECT_MIN,
    ASPECT_MAX,
    MIN_FRONTALITY,
)
from app.utils.landmarks import (
    valid_landmarks,
    valid_landmarks_strict,
    frontality_score,
)


def detect_and_filter(detector, frame):
    """Run RetinaFace + apply two-stage filter.

    Returns:
        kept_meta: list of (box_int, lmks, det_score, frontality, identify_ok)
        tracker_arr: np.ndarray [N, 5] (x1,y1,x2,y2,score) — ByteTracker input
    """
    if DETECT_SCALE < 1.0:
        det_input = cv2.resize(frame, None, fx=DETECT_SCALE, fy=DETECT_SCALE)
    else:
        det_input = frame

    with torch.no_grad():
        dets = detector.detect_faces(det_input, conf_threshold=DET_THRESH, nms_threshold=NMS_THRESH)

    if dets is not None and len(dets) > 0 and DETECT_SCALE < 1.0:
        dets = dets.copy()
        inv = 1.0 / DETECT_SCALE
        dets[:, :4] *= inv
        dets[:, 5:15] *= inv

    kept_meta = []
    tracker_input = []
    if dets is not None and len(dets) > 0:
        for det in dets:
            box = det[0:4].astype(int)
            lmks = det[5:15].reshape(5, 2).astype(np.float32)
            score = float(det[4])
            w, h = box[2] - box[0], box[3] - box[1]

            # --- Stage 1: pre-FLOW2 loose gates feed ByteTrack ---
            # Same filter set as main branch had: size + aspect + LOOSE landmarks.
            # Keeps tracker behavior identical to pre-FLOW2 so occlusion / profile
            # / hand-on-face frames still maintain track continuity.
            if min(w, h) < MIN_FACE_SIZE:
                continue
            if not (ASPECT_MIN < w / max(1, h) < ASPECT_MAX):
                continue
            if not valid_landmarks(lmks, box):
                continue
            tracker_input.append([float(box[0]), float(box[1]), float(box[2]), float(box[3]), score])

            # --- Stage 2: strict gates decide identify_ok (only affects AdaFace path) ---
            fscore = frontality_score(lmks)
            identify_ok = True
            if not valid_landmarks_strict(lmks, box):
                identify_ok = False
            # [FLOW2-RISK #8] hard reject anything below MIN_FRONTALITY for ID,
            # but the det still went to ByteTrack so track survives.
            if identify_ok and fscore < MIN_FRONTALITY:
                identify_ok = False

            kept_meta.append((box, lmks, score, fscore, identify_ok))

    tracker_arr = (np.asarray(tracker_input, dtype=np.float32)
                   if tracker_input else np.empty((0, 5), dtype=np.float32))
    return kept_meta, tracker_arr
