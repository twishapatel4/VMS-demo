import os
import time
import cv2
import torch
import numpy as np
import threading
from collections import deque

os.environ["QT_QPA_PLATFORM"] = "xcb"

from retinaface.retinaface import RetinaFace
from net import build_model
from osnet_arch import osnet_x0_25
from bytetrack import BYTETracker

# --- SPEED OPTIMIZATION ---
torch.backends.cudnn.benchmark = True

# --- PATHS ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, 'weights')
ADAFACE_CHECKPOINT = os.path.join(WEIGHTS_DIR, 'adaface_ir101_ms1mv2.ckpt')
# RETINEFACE_CHECKPOINT = os.path.join(WEIGHTS_DIR, 'detection_mobilenet0.25_Final.pth')
RETINEFACE_CHECKPOINT = os.path.join(WEIGHTS_DIR, 'detection_Resnet50_Final.pth')  # Updated checkpoint name
OSNET_CHECKPOINT = os.path.join(WEIGHTS_DIR, 'osnet_x0_25_imagenet.pth')

# --- SETTINGS ---
EMB_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'embeddings')
KNOWN_DIR = os.path.join(EMB_ROOT, 'known')
UNKNOWN_DIR = os.path.join(EMB_ROOT, 'unknown')
THRESHOLD = 0.35
DET_THRESH = 0.70           # RetinaFace confidence - surity this is a face or not
NMS_THRESH = 0.25          # merge multiple boundary boxes to one
MIN_FACE_SIZE = 30          # px, reject tiny faces
SKIP_FRAMES = 2              # for speed: only run detection/recognition every N frames
DETECTION_SCALE=1.2
OSNET_THRESH = 0.80          # body ReID cosine match threshold

# --- DETECTION PERF / FILTER GATES ---
DETECT_SCALE = 0.5           # downscale frame before RetinaFace forward; boxes scaled back. 1.0 = no downscale.
ASPECT_MIN = 0.45            # min bbox W/H (rejects vertical/elongated body parts)
ASPECT_MAX = 1.4             # max bbox W/H (rejects horizontal/elongated body parts)
TOPK_GALLERY = 1             # max stored embeddings per gallery entry (top-K matching)

# --- LOGGING ---
STATS_EVERY_FRAMES = 60      # print FPS + gallery stats every N frames

# --- CAMERA SOURCES ---
# (label, src) — label shown on window + logs. src can be int (local) or URL (remote stream).
CAM_SOURCES = [
    ("Laptop",  0),
    ("Webcam",  "http://192.168.29.97:5000/video"),
]

# --- SHORT-TERM MEMORY (Temporary Unknown Gallery) ---
UNKNOWN_THRESH = 0.42     # cross-frame stranger match; lowered because EMA centroid is cleaner than raw embed
UNKNOWN_TTL_SECONDS = 7200   # 2 hours
UNKNOWN_EMA_ALPHA = 0.3      # running avg: new_feat * alpha + old_feat * (1-alpha); 0 = no update

# --- TRACK-LEVEL VOTING (K=1 gallery, smarter query) ---
TRACK_QUERY_BUFFER = 4           # embeds collected per new track before voting against gallery
REEMBED_EVERY_FRAMES = 15        # how often a bound track contributes a fresh embed (EMA update)
GALLERY_WRITE_MIN_SCORE = 0.90   # det score required to write to gallery (quality gate)
GALLERY_WRITE_MIN_SIZE = 50      # px min face size for gallery writes

# --- BYTETRACK (per-cam motion tracker; binds local track_id -> label) ---
BYTETRACK_TRACK_THRESH = 0.5     # min det score to confirm a new track
BYTETRACK_MATCH_THRESH = 0.8     # IoU cost upper bound for first association
BYTETRACK_BUFFER = 90            # frames to keep lost tracks before removal (~3s @ 30fps)
BYTETRACK_FRAME_RATE = 30        # nominal FPS for buffer scaling
TRACK_DET_IOU_MIN = 0.3          # min IoU to associate a track to a detection for embedding

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[SYSTEM] PyTorch Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"[SYSTEM] GPU: {torch.cuda.get_device_name(0)}")

# --- GLOBAL DATABASE CACHE ---
KNOWN_EMBS = None
KNOWN_NAMES = []


def reload_database():
    """Walk KNOWN_DIR/*.npy. Each file = one person; name = filename stem.
    File may store (1, 512) or (K, 512) — take mean, L2-normalize, stack."""
    global KNOWN_EMBS, KNOWN_NAMES
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
        KNOWN_EMBS = torch.from_numpy(np.stack(rows, axis=0)).to(DEVICE).float()
        KNOWN_NAMES = names
        print(f"[CACHE] Database loaded: {len(KNOWN_NAMES)} people from {KNOWN_DIR}")
    else:
        KNOWN_EMBS = None
        KNOWN_NAMES = []


reload_database()


# --- LOAD MODELS ---
def load_adaface(path):
    model = build_model('ir_101')
    try:
        checkpoint = torch.load(path, map_location=DEVICE)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        model_dict = model.state_dict()
        new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()
                          if k.replace('model.', '') in model_dict}
        model.load_state_dict(new_state_dict, strict=False)
        model.to(DEVICE)
        # model.half()  # Use half precision for faster inference on compatible GPUs
        model.eval()
        return model
    except Exception as e:
        print(f"[FATAL] AdaFace Load Error: {e}")
        exit()


def load_retinaface(path, network='mobile0.25', half=False):
    model = RetinaFace(network_name=network, half=half, device=DEVICE)
    try:
        state = torch.load(path, map_location=lambda s, l: s)
        # strip DataParallel 'module.' prefix if present
        state = {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        model.eval()
        model = model.to(DEVICE)
        return model
    except Exception as e:
        print(f"[FATAL] RetinaFace Load Error: {e}")
        exit()


# RetinaFace mobile (PyTorch, no onnxruntime). Loaded from explicit weights path.
# detector = load_retinaface(RETINEFACE_CHECKPOINT, network='mobile0.25', half=False)
detector = load_retinaface(RETINEFACE_CHECKPOINT, network='resnet50', half=False)

adaface = load_adaface(ADAFACE_CHECKPOINT)


OSNET_INPUT_H, OSNET_INPUT_W = 256, 128
OSNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
OSNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)


def load_osnet():
    if not os.path.exists(OSNET_CHECKPOINT):
        print(f"[WARN] OSNet checkpoint missing at {OSNET_CHECKPOINT}. Body ReID disabled.")
        return None
    try:
        model = osnet_x0_25(num_classes=1000, pretrained=False, loss='softmax')
        ckpt = torch.load(OSNET_CHECKPOINT, map_location=DEVICE)
        state = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
        # Strip classifier head — we only need features in eval mode.
        state = {k: v for k, v in state.items() if not k.startswith('classifier')}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            print(f"[WARN] OSNet unexpected keys: {len(unexpected)} (ignored)")
        model.eval().to(DEVICE)
        print("[SYSTEM] OSNet loaded (body ReID active).")
        return model
    except Exception as e:
        print(f"[WARN] OSNet load failed: {e}. Body ReID disabled.")
        return None


osnet = load_osnet()


def verify_gpu_placement():
    """Print device + dtype for every loaded model. Confirms GPU placement."""
    print("[INIT] ============ GPU PLACEMENT CHECK ============")
    print(f"[INIT] DEVICE = {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"[INIT] GPU    = {torch.cuda.get_device_name(0)}")
        print(f"[INIT] CUDA   = {torch.version.cuda}, cuDNN = {torch.backends.cudnn.version()}")

    def _info(name, model):
        if model is None:
            print(f"[INIT]   {name:<12s} NOT LOADED")
            return
        try:
            p = next(model.parameters())
            print(f"[INIT]   {name:<12s} device={p.device}  dtype={p.dtype}  params={sum(x.numel() for x in model.parameters())/1e6:.2f}M")
        except StopIteration:
            print(f"[INIT]   {name:<12s} (no parameters)")

    _info("RetinaFace", detector)
    _info("AdaFace",    adaface)
    _info("OSNet",      osnet)

    if KNOWN_EMBS is not None:
        print(f"[INIT]   KNOWN_EMBS   device={KNOWN_EMBS.device}  dtype={KNOWN_EMBS.dtype}  N={KNOWN_EMBS.shape[0]}")
    else:
        print(f"[INIT]   KNOWN_EMBS   None (empty DB)")

    if DEVICE.type == 'cuda':
        alloc = torch.cuda.memory_allocated() / 1e9
        resv  = torch.cuda.memory_reserved()  / 1e9
        print(f"[INIT]   GPU mem      allocated={alloc:.2f}GB  reserved={resv:.2f}GB")
    print("[INIT] =============================================")


verify_gpu_placement()


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
    """Geometric sanity check on 5 landmarks (R-eye, L-eye, nose, R-mouth, L-mouth) vs bbox.
    Rejects garbage landmark layouts produced when detector fires on non-faces (elbow, fist, etc.)."""
    x1, y1, x2, y2 = box
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    margin = 0.05 * max(w, h)

    # All landmarks must sit (mostly) inside the bbox.
    if (lmks[:, 0].min() < x1 - margin or lmks[:, 0].max() > x2 + margin or
        lmks[:, 1].min() < y1 - margin or lmks[:, 1].max() > y2 + margin):
        return False

    # Eye separation should be a meaningful fraction of face width.
    eye_dist = float(np.linalg.norm(lmks[0] - lmks[1]))
    if eye_dist < 0.15 * w:
        return False

    # Eyes must be above mouth corners (image y increases downward).
    if lmks[0, 1] > lmks[3, 1] or lmks[1, 1] > lmks[4, 1]:
        return False

    return True


def body_crop(frame, box, scale_h=3.5, scale_w=1.5):
    """Expand face box downward to estimate full-body region."""
    x1, y1, x2, y2 = box
    fh, fw = y2 - y1, x2 - x1
    cx = (x1 + x2) // 2
    new_w = int(fw * scale_w)
    bx1 = max(0, cx - new_w // 2)
    bx2 = min(frame.shape[1], cx + new_w // 2)
    by1 = max(0, y1 - int(fh * 0.2))
    by2 = min(frame.shape[0], y1 + int(fh * scale_h))
    crop = frame[by1:by2, bx1:bx2]
    return crop if crop.size > 0 else None


def osnet_embed(crops_bgr):
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


is_naming = False


def admin_input_thread():
    global is_naming
    try:
        snap = unknown_gallery.snapshot()
        if not snap:
            print("\n[REGISTER] No unknowns in gallery. Nothing to register.")
            return
        visible = ", ".join(f"#{eid:03d}(seen={info['count']})" for eid, info in sorted(snap.items()))
        print(f"\n[REGISTER] Active unknowns: {visible}")
        print("[REGISTER] Type ID to register, 'clear' to wipe gallery, 'cancel' to abort.")
        raw = input(">>> ENTER UNKNOWN ID: ").strip(" \t\n\r​﻿ ")
        if raw.lower() in ('cancel', 'c', 'q', ''):
            print("[REGISTER] Cancelled.")
            return
        if raw.lower() == 'clear':
            n = unknown_gallery.clear()
            body_gallery.clear()
            wipe_unknown_dir()
            for cs in cam_states.values():
                cs.track_to_label.clear()
            print(f"[GALLERY] Cleared {n} unknown entries (RAM + disk). IDs reset to 001.")
            return
        try:
            eid = int(raw)
        except ValueError:
            print(f"[REGISTER] Invalid ID input: {repr(raw)}. Cancelled.")
            return
        feat = unknown_gallery.get_feat(eid)
        if feat is None:
            print(f"[REGISTER] ID #{eid:03d} not found. Cancelled.")
            return
        name = input(">>> ENTER NAME: ").strip()
        if not name:
            print("[REGISTER] Empty name. Cancelled.")
            return
        save_to_db(name, feat)
        unknown_gallery.remove(eid)
        body_gallery.remove(eid)
        delete_unknown_file(eid)
    finally:
        is_naming = False


def embed_batch(aligned_batch):
    # aligned_batch: tensor [N, 3, 112, 112] in [0,255], RGB.
    x = (aligned_batch - 127.5) / 128.0
    x = x.to(DEVICE)  # Use half precision for faster inference on compatible GPUs
    # .half()
    with torch.no_grad():
        feats, _ = adaface(x)
    feats = feats.float()  # cast back to fp32 for DB matmul (KNOWN_EMBS is fp32)
    feats = feats / (torch.norm(feats, dim=1, keepdim=True) + 1e-8)
    return feats


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

    def assign(self, feat, now):
        """Returns (eid, sim). Reuses entry (EMA-blend slot) if match, else creates new."""
        with self.lock:
            best_eid, best_sim = self._best_match_unsafe(feat)
            if best_eid is not None and best_sim >= self.match_thresh:
                self._ema_blend_unsafe(best_eid, feat, now)
                return best_eid, best_sim
            eid = self.next_id
            self.next_id += 1
            self.entries[eid] = {
                'embeds': deque([feat.detach().clone()], maxlen=self.maxlen),
                'last_seen': now,
                'count': 1,
            }
            return eid, 0.0

    def _ema_blend_unsafe(self, eid, feat, now):
        """Caller must hold lock. K=1 EMA: blend fresh embed into single stored slot, re-normalize."""
        e = self.entries[eid]
        if e['embeds']:
            old = e['embeds'][0]
            blended = UNKNOWN_EMA_ALPHA * feat + (1.0 - UNKNOWN_EMA_ALPHA) * old
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

    def update_entry(self, eid, feat, now):
        """EMA-blend fresh embed into stored slot. Creates entry if absent."""
        with self.lock:
            if eid in self.entries:
                self._ema_blend_unsafe(eid, feat, now)
            else:
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


unknown_gallery = UnknownGallery()
body_gallery = UnknownGallery(match_thresh=OSNET_THRESH, ttl=UNKNOWN_TTL_SECONDS, maxlen=TOPK_GALLERY)


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


load_unknowns_from_disk()


class VideoStream:
    def __init__(self, src=0, label="cam", width=1280, height=720):
        self.label = label
        self.src = src
        self.cap = cv2.VideoCapture(src)
        # Resolution hints only meaningful for local devices; remote streams ignore.
        if isinstance(src, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.opened = self.cap.isOpened()
        self.ret, self.frame = (False, None)
        if self.opened:
            self.ret, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        if not self.opened:
            print(f"[WARN] {self.label}: failed to open source {self.src}")
            return self
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        if self.opened:
            self.cap.release()


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


def _iou_xyxy(a, b):
    xx1 = max(a[0], b[0]); yy1 = max(a[1], b[1])
    xx2 = min(a[2], b[2]); yy2 = min(a[3], b[3])
    w = max(0.0, xx2 - xx1); h = max(0.0, yy2 - yy1)
    inter = w * h
    area_a = max(0.0, (a[2] - a[0])) * max(0.0, (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / max(union, 1e-6)


def process_frame(frame, now, cam_state, frame_idx):
    """Detect → ByteTrack → (a) bound tracks: periodic EMA gallery update.
                            (b) unbound tracks: accumulate TRACK_QUERY_BUFFER embeds, vote, bind."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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

    kept_meta = []      # (box_int, lmks, det_score)
    tracker_input = []
    if dets is not None and len(dets) > 0:
        for det in dets:
            box = det[0:4].astype(int)
            lmks = det[5:15].reshape(5, 2).astype(np.float32)
            score = float(det[4])
            w, h = box[2] - box[0], box[3] - box[1]
            if min(w, h) < MIN_FACE_SIZE:
                continue
            if not (ASPECT_MIN < w / max(1, h) < ASPECT_MAX):
                continue
            if not valid_landmarks(lmks, box):
                continue
            kept_meta.append((box, lmks, score))
            tracker_input.append([float(box[0]), float(box[1]), float(box[2]), float(box[3]), score])

    tracker_arr = (np.asarray(tracker_input, dtype=np.float32)
                   if tracker_input else np.empty((0, 5), dtype=np.float32))
    tracks = cam_state.tracker.update(tracker_arr)

    new_results = []
    pending_crops = []
    pending_tags = []   # ("UPDATE", tid, eid) | ("VOTE", tid, track_box, det_box)

    for st in tracks:
        tid = st.track_id
        track_box = st.tlbr.astype(int)

        # Best-IoU det for this track (used for both UPDATE quality gate and VOTE crop).
        best_i, best_iou = -1, 0.0
        for i, (b, _, _) in enumerate(kept_meta):
            v = _iou_xyxy(track_box, b)
            if v > best_iou:
                best_iou = v
                best_i = i

        # Branch A: tid already labeled → reuse + maybe contribute fresh embed to EMA.
        if tid in cam_state.track_to_label:
            name, color = cam_state.track_to_label[tid]
            new_results.append((track_box, name, color))

            eid = cam_state.track_to_eid.get(tid)
            if eid is None or best_i < 0 or best_iou < TRACK_DET_IOU_MIN:
                continue

            last = cam_state.track_last_embed.get(tid, -10**9)
            if frame_idx - last < REEMBED_EVERY_FRAMES:
                continue

            det_box, lmks, score = kept_meta[best_i]
            w = det_box[2] - det_box[0]
            h = det_box[3] - det_box[1]
            if score < GALLERY_WRITE_MIN_SCORE or min(w, h) < GALLERY_WRITE_MIN_SIZE:
                continue

            pending_crops.append(norm_crop(rgb, lmks, size=112))
            pending_tags.append(("UPDATE", tid, eid))
            cam_state.track_last_embed[tid] = frame_idx
            continue

        # Branch B: tid not yet bound → need TRACK_QUERY_BUFFER embeds before voting.
        if best_i < 0 or best_iou < TRACK_DET_IOU_MIN:
            new_results.append((track_box, "...", (200, 200, 200)))
            continue

        det_box, lmks, _ = kept_meta[best_i]
        pending_crops.append(norm_crop(rgb, lmks, size=112))
        pending_tags.append(("VOTE", tid, track_box, det_box))

    if pending_crops:
        batch_np = np.stack(pending_crops, axis=0)
        batch = torch.from_numpy(batch_np).permute(0, 3, 1, 2).contiguous().to(DEVICE)
        feats = embed_batch(batch)

        for i, tag in enumerate(pending_tags):
            feat = feats[i:i+1]

            if tag[0] == "UPDATE":
                _, tid, eid = tag
                unknown_gallery.update_entry(eid, feat, now)
                persist_unknown(eid)
                continue

            # VOTE — accumulate; only resolve once buffer full.
            _, tid, track_box, det_box = tag
            buf = cam_state.track_query_buf.setdefault(tid, [])
            buf.append(feat)

            if len(buf) < TRACK_QUERY_BUFFER:
                new_results.append((track_box, "...", (200, 200, 200)))
                continue

            stack = torch.cat(buf, dim=0)
            mean_feat = stack.mean(dim=0, keepdim=True)
            mean_feat = mean_feat / (torch.norm(mean_feat, dim=1, keepdim=True) + 1e-8)
            cam_state.track_query_buf.pop(tid, None)

            name = "Unknown"
            color = (0, 0, 255)
            assigned_eid = None
            matched_known = False

            if KNOWN_EMBS is not None and len(KNOWN_NAMES) > 0:
                sims = torch.mm(mean_feat, KNOWN_EMBS.t())
                max_val, max_idx = torch.max(sims, dim=1)
                if max_val.item() >= THRESHOLD:
                    name = f"{KNOWN_NAMES[max_idx.item()]} ({max_val.item():.2f})"
                    color = (0, 255, 0)
                    matched_known = True

            if not matched_known:
                face_eid, _ = unknown_gallery.find_match(mean_feat)
                if face_eid is not None:
                    unknown_gallery.update_entry(face_eid, mean_feat, now)
                    assigned_eid = face_eid
                else:
                    bfeat = None
                    if osnet is not None:
                        bcrop = body_crop(frame, det_box)
                        if bcrop is not None:
                            bfeat = osnet_embed([bcrop])
                            body_eid, _ = body_gallery.find_match(bfeat)
                            if body_eid is not None:
                                assigned_eid = body_eid
                                unknown_gallery.update_entry(assigned_eid, mean_feat, now)
                                body_gallery.update_entry(assigned_eid, bfeat, now)
                    if assigned_eid is None:
                        assigned_eid, _ = unknown_gallery.assign(mean_feat, now)
                        if bfeat is not None:
                            body_gallery.update_entry(assigned_eid, bfeat, now)
                persist_unknown(assigned_eid)
                name = f"Unknown_{assigned_eid:03d}"
                color = (0, 0, 255)

            cam_state.track_to_label[tid] = (name, color)
            if assigned_eid is not None:
                cam_state.track_to_eid[tid] = assigned_eid
            cam_state.track_last_embed[tid] = frame_idx
            new_results.append((track_box, name, color))

    # Prune all per-track state to alive tids.
    alive = {st.track_id for st in tracks}
    cam_state.track_to_label   = {t: v for t, v in cam_state.track_to_label.items()   if t in alive}
    cam_state.track_to_eid     = {t: v for t, v in cam_state.track_to_eid.items()     if t in alive}
    cam_state.track_query_buf  = {t: v for t, v in cam_state.track_query_buf.items()  if t in alive}
    cam_state.track_last_embed = {t: v for t, v in cam_state.track_last_embed.items() if t in alive}

    return new_results


# --- MAIN LOOP ---
streams = [VideoStream(src=src, label=label, width=1280, height=720).start()
           for label, src in CAM_SOURCES]
active_streams = [vs for vs in streams if vs.opened]
if not active_streams:
    print("[FATAL] No cameras opened. Exiting.")
    raise SystemExit(1)

for vs in active_streams:
    w = vs.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = vs.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"[SYSTEM] {vs.label} resolution: {w}x{h}")

# Per-stream state. Shared unknown_gallery → same person cross-cam = same Unknown_XXX.
cam_states = {vs.label: CamState(vs.label) for vs in active_streams}
frame_counts = {vs.label: 0 for vs in active_streams}
last_results_map = {vs.label: [] for vs in active_streams}
last_evict = 0.0

# FPS stats
stats_start = time.time()
stats_total_frames = 0

print(f"[SYSTEM] Loop starting with {len(active_streams)} cam(s) at target 20+ FPS...")

while True:
    now = time.time()
    if now - last_evict > 5.0:
        unknown_gallery.evict(now)
        body_gallery.evict(now)
        last_evict = now

    for vs in active_streams:
        frame = vs.read()
        if frame is None:
            continue

        frame_counts[vs.label] += 1
        stats_total_frames += 1

        if frame_counts[vs.label] % SKIP_FRAMES == 0:
            last_results_map[vs.label] = process_frame(frame, now, cam_states[vs.label], frame_counts[vs.label])

        # Draw every frame for smoothness.
        for box, name, color in last_results_map[vs.label]:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(frame, name, (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow(f"VMS - {vs.label}", frame)

    # Periodic FPS + gallery stats.
    if stats_total_frames >= STATS_EVERY_FRAMES:
        dt = time.time() - stats_start
        fps = stats_total_frames / max(dt, 1e-6)
        gpu_mem = ""
        if DEVICE.type == 'cuda':
            gpu_mem = f" gpu_mem={torch.cuda.memory_allocated()/1e9:.2f}GB"
        # print(f"[STATS] frames={stats_total_frames} dt={dt:.1f}s fps={fps:.1f} face_gal={len(unknown_gallery.entries)} body_gal={len(body_gallery.entries)}{gpu_mem}")
        stats_start = time.time()
        stats_total_frames = 0

    key = cv2.waitKey(1) & 0xFF
    if key == ord('e') and not is_naming:
        is_naming = True
        threading.Thread(target=admin_input_thread, daemon=True).start()
    elif key == ord('c'):
        n = unknown_gallery.clear()
        body_gallery.clear()
        wipe_unknown_dir()
        for cs in cam_states.values():
            cs.track_to_label.clear()
        print(f"[GALLERY] Cleared {n} unknown entries (RAM + disk). IDs reset to 001.")
    elif key == ord('q'):
        persist_all_unknowns()
        break

for vs in streams:
    vs.stop()
cv2.destroyAllWindows()