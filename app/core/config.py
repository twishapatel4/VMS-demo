"""Central configuration for VMS face-recognition pipeline.

All tunable constants live here. FLOW2 risk comments preserved verbatim from
the legacy monolithic main.py so revert/audit semantics stay intact.
"""

import os
import torch

# --- PATHS ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, 'weights')
ADAFACE_CHECKPOINT = os.path.join(WEIGHTS_DIR, 'adaface_ir101_ms1mv2.ckpt')
# RETINEFACE_CHECKPOINT = os.path.join(WEIGHTS_DIR, 'detection_mobilenet0.25_Final.pth')
RETINEFACE_CHECKPOINT = os.path.join(WEIGHTS_DIR, 'detection_Resnet50_Final.pth')  # Updated checkpoint name
OSNET_CHECKPOINT = os.path.join(WEIGHTS_DIR, 'osnet_x0_25_imagenet.pth')

EMB_ROOT = os.path.join(PROJECT_ROOT, 'embeddings')
KNOWN_DIR = os.path.join(EMB_ROOT, 'known')
UNKNOWN_DIR = os.path.join(EMB_ROOT, 'unknown')
BODY_DIR = os.path.join(EMB_ROOT, 'body')   # NEW: body ReID embeds persisted to disk (FLOW2 hybrid)

# --- DETECTION / MATCH THRESHOLDS ---
THRESHOLD = 0.35
DET_THRESH = 0.30           # RetinaFace confidence - surity this is a face or not
NMS_THRESH = 0.25           # merge multiple boundary boxes to one
MIN_FACE_SIZE = 20          # px, reject tiny faces
SKIP_FRAMES = 1             # main loop: call process_frame every N frames (1 = every frame)

# === [FLOW2-RISK #4] identify-every-2nd-frame ==============================
# CHANGE : detection + ByteTrack still run every frame, but AdaFace embed +
#          cascade match + Branch B bind only fire every IDENTIFY_INTERVAL frames.
#          Branch A label cache draws on skip frames so motion stays smooth.
# REASON : ~50% AdaFace compute saved per FLOW2.
# REVERT : set IDENTIFY_INTERVAL = 1
# ACCURACY: new tid first appearing on skip frame has no label until next
#          identify frame (~33ms). No drop on already-bound tracks.
# ===========================================================================
IDENTIFY_INTERVAL = 2

DETECTION_SCALE = 1.2
OSNET_THRESH = 0.80          # body ReID cosine match threshold (kept current; FLOW2 said 0.60)

# === [FLOW2-RISK #7] body-crop-validity-gate ===============================
# CHANGE : body_crop now returns None unless crop.w >= BODY_CROP_MIN_W and
#          crop.h >= BODY_CROP_MIN_H. Drives FLOW2 unknown path left/right gate.
# REASON : tiny crops give noisy OSNet embeds → bad body matches → wrong reuse.
# REVERT : set BODY_CROP_MIN_W = 0, BODY_CROP_MIN_H = 0
# ACCURACY: distant subjects route to face-only branch. Alice-fix
#          (face DB cross-check on body NO MATCH) catches the duplicate case.
# ===========================================================================
BODY_CROP_MIN_W = 80
BODY_CROP_MIN_H = 40

# === [FLOW2-NEW] secondary-face-check-threshold ============================
# Used when body_gallery matches (≥ OSNET_THRESH). Verifies the face embed
# stored at that eid also matches the current face embed at ≥ this threshold.
# Prevents identity pollution from same-clothes false body matches.
# REVERT : set very low (e.g. 0.0) to effectively disable the check.
# ===========================================================================
SECONDARY_FACE_THRESH = 0.35

# --- DETECTION PERF / FILTER GATES ---
DETECT_SCALE = 1.0           # downscale frame before RetinaFace forward; boxes scaled back. 1.0 = no downscale.
ASPECT_MIN = 0.30            # min bbox W/H (lowered to admit profile/side faces)
ASPECT_MAX = 1.4             # max bbox W/H (rejects horizontal/elongated body parts)
TOPK_GALLERY = 1             # max stored embeddings per gallery entry (top-K matching)

# --- FRONTALITY SOFT-WEIGHTING ---
# Frontality scores in [0, 1]: 1.0 = frontal, 0.0 = full profile.
# Used to scale how much each frame's embed contributes to centroid / vote.
# Not a hard gate — profile frames still contribute, just less.
FRONTAL_REF = 0.55           # frontality at/above this → full weight (1.0)
FRONTAL_WEIGHT_FLOOR = 0.15  # weight cannot drop below this (profile still nudges centroid)

# === [FLOW2-RISK #8] frontality-hard-reject + landmark-geometry-sanity ======
# CHANGE : detections with frontality_score < MIN_FRONTALITY are rejected
#          outright (not just weighted). valid_landmarks() also enforces
#          stricter geometric sanity (nose between eyes horizontally + between
#          eye/mouth lines vertically + mouth/eye width ratio in [0.4, 1.8]).
# REASON : RetinaFace fires on back-of-head / hair patches and fabricates
#          5 landmarks. Without these gates, those false positives produce
#          look-alike low-info embeds that cluster into the same Unknown_NNN
#          eid — different people getting the same ID.
# REVERT : set MIN_FRONTALITY = 0.0 and remove the extra geometry checks
#          inside valid_landmarks(). Soft frontality weighting still works.
# ACCURACY: back-of-head / extreme profile rejected → fewer bogus eids.
#          True 3/4-profile faces (frontality ~0.25-0.40) still admitted.
#          Risk: very extreme side profiles (frontality < 0.20) now produce
#          no label until person turns more toward camera.
# ===========================================================================
MIN_FRONTALITY = 0.20

# --- LOGGING ---
STATS_EVERY_FRAMES = 60      # print FPS + gallery stats every N frames

# --- CAMERA SOURCES ---
# (label, src) — label shown on window + logs. src can be int (local) or URL (remote stream).
CAM_SOURCES = [
    ("Laptop",  0),
    # ("Webcam",  "http://192.168.29.97:5000/video"), #ArpanBhai
    ("Webcam1",  "http://192.168.29.218:5000/video"), # JeelBhai~
]

# --- SHORT-TERM MEMORY (Temporary Unknown Gallery) ---
# === [FLOW2-RISK #5] face-fallback-threshold ===============================
# CHANGE : UNKNOWN_THRESH lowered 0.42 → 0.40 to match FLOW2 face-fallback spec.
# REASON : FLOW2 alignment.
# REVERT : set UNKNOWN_THRESH = 0.42
# ACCURACY: small step (0.02); minor risk of cross-person reuse at borderline.
# ===========================================================================
UNKNOWN_THRESH = 0.40        # cross-frame stranger match (FLOW2 hybrid)
UNKNOWN_TTL_SECONDS = 7200   # 2 hours
UNKNOWN_EMA_ALPHA = 0.3      # running avg: new_feat * alpha + old_feat * (1-alpha); 0 = no update

# --- TRACK-LEVEL VOTING (K=1 gallery, smarter query) ---
TRACK_QUERY_BUFFER = 3        # [FLOW2-RISK #1] embeds collected per new track before weighted-mean vote. Was 6 pre-FLOW2; lowered to 3 for snappier bind while still filtering bad first frames. Set 1 = first-frame bind, 6 = old behavior.
REEMBED_EVERY_FRAMES = 15        # how often a bound track contributes a fresh embed (EMA update)
GALLERY_WRITE_MIN_SCORE = 0.90   # det score required to write to gallery (quality gate)
GALLERY_WRITE_MIN_SIZE = 40      # px min face size for gallery writes

# --- BYTETRACK (per-cam motion tracker; binds local track_id -> label) ---
BYTETRACK_TRACK_THRESH = 0.30     # min det score to confirm a new track
# === [FLOW2-RISK #10] fast-motion tracking ================================
# CHANGE : MATCH_THRESH 0.8 -> 0.9  (more permissive IoU first-pass association)
#          BUFFER 90 -> 180         (lost track survives 6s vs 3s)
# REASON : fast head motion makes Kalman prediction overshoot; default IoU
#          + 3s buffer drop the labeled track too fast, causing the
#          green->red->gray->green flicker on known persons.
# REVERT : set MATCH_THRESH = 0.8 and BUFFER = 90.
# ACCURACY: track holds longer through fast motion. Small crowd-swap risk
#          if two faces overlap heavily — mitigated by Stage 2 identify_ok
#          + stitching IoU gate.
# ==========================================================================
BYTETRACK_MATCH_THRESH = 0.9     # was 0.8 — see [FLOW2-RISK #10]
BYTETRACK_BUFFER = 180           # was 90 — see [FLOW2-RISK #10]
BYTETRACK_FRAME_RATE = 30        # nominal FPS for buffer scaling
TRACK_DET_IOU_MIN = 0.3          # min IoU to associate a track to a detection for embedding

# === [FLOW2-RISK #11] track stitching =====================================
# When ByteTrack does drop a labeled tid (e.g. >6s of no usable detection)
# and reassigns a fresh tid for the same person, recover the label without
# running a 3-frame vote. Cache labeled bboxes; new tid spatially overlapping
# a recently-lost label inherits it.
# REASON : prevents green->red->gray flicker when ByteTrack reassigns tid.
# REVERT : set STITCH_WINDOW_FRAMES = 0 (effectively disables match).
# ACCURACY: rebind is instant + correct in 90%+ cases. Risk: another person
#          walking into the spot within STITCH_WINDOW frames inherits old
#          label. Mitigated by IoU gate.
# ==========================================================================
STITCH_WINDOW_FRAMES = 120       # ~4s @ 30fps; cache valid this long after tid lost
STITCH_IOU_MIN = 0.3             # bbox overlap required between current track and cached entry

# --- DEVICE ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
