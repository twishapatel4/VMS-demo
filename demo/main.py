import os
import time
import cv2
import torch
import numpy as np
import threading

os.environ["QT_QPA_PLATFORM"] = "xcb"

from retinaface.retinaface import RetinaFace
from net import build_model

# --- SPEED OPTIMIZATION ---
torch.backends.cudnn.benchmark = True

# --- PATHS ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, 'weights')
ADAFACE_CHECKPOINT = os.path.join(WEIGHTS_DIR, 'adaface_ir101_ms1mv2.ckpt')
RETINEFACE_CHECKPOINT = os.path.join(WEIGHTS_DIR, 'detection_mobilenet0.25_Final.pth')

# --- SETTINGS ---
EMB_DB_PATH = 'vms_embeddings.npy'
NAME_DB_PATH = 'vms_names.txt'
THRESHOLD = 0.35
DET_THRESH = 0.50           # RetinaFace confidence - surity this is a face or not
NMS_THRESH = 0.40           # merge multiple boundary boxes to one
MIN_FACE_SIZE = 30          # px, reject tiny faces
SKIP_FRAMES = 2              # for speed: only run detection/recognition every N frames
DETECTION_SCALE=1.2

# --- CAMERA SOURCES ---
# (label, src) — label shown on window + logs. src can be int (local) or URL (remote stream).
CAM_SOURCES = [
    ("Laptop",  0),
    ("Webcam",  "http://192.168.29.97:5000/video"),
]

# --- SHORT-TERM MEMORY (Temporary Unknown Gallery) ---
UNKNOWN_THRESH = 0.40     # cross-frame stranger match; stricter than DB THRESHOLD
UNKNOWN_TTL_SECONDS = 7200   # 2 hours
UNKNOWN_EMA_ALPHA = 0.3      # running avg: new_feat * alpha + old_feat * (1-alpha); 0 = no update

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[SYSTEM] PyTorch Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"[SYSTEM] GPU: {torch.cuda.get_device_name(0)}")

# --- GLOBAL DATABASE CACHE ---
KNOWN_EMBS = None
KNOWN_NAMES = []


def reload_database():
    global KNOWN_EMBS, KNOWN_NAMES
    if os.path.exists(EMB_DB_PATH) and os.path.exists(NAME_DB_PATH):
        try:
            embs = np.load(EMB_DB_PATH)
            with open(NAME_DB_PATH, 'r') as f:
                names = f.read().splitlines()

            if len(embs) == len(names) and len(names) > 0:
                KNOWN_EMBS = torch.from_numpy(embs).to(DEVICE).float()
                KNOWN_NAMES = names
                print(f"[CACHE] Database loaded successfully: {len(KNOWN_NAMES)} people.")
            else:
                print(f"[WARN] Database mismatch: {len(embs)} vectors vs {len(names)} names. Recognition disabled.")
                KNOWN_EMBS = None
                KNOWN_NAMES = []
        except Exception as e:
            print(f"[CACHE ERROR] {e}")
            KNOWN_EMBS = None


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
        model.half()  # Use half precision for faster inference on compatible GPUs
        model.eval()
        return model
    except Exception as e:
        print(f"[FATAL] AdaFace Load Error: {e}")
        exit()


def load_retinaface(path, network='mobile0.25', half=True):
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
detector = load_retinaface(RETINEFACE_CHECKPOINT, network='mobile0.25', half=True)

adaface = load_adaface(ADAFACE_CHECKPOINT)

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


def save_to_db(name, vector):
    vector_np = vector.cpu().numpy().flatten() if torch.is_tensor(vector) else vector.flatten()

    if os.path.exists(EMB_DB_PATH):
        db = np.load(EMB_DB_PATH)
        db = np.vstack([db, vector_np])
    else:
        db = vector_np.reshape(1, -1)

    np.save(EMB_DB_PATH, db)
    with open(NAME_DB_PATH, 'a') as f:
        f.write(name + "\n")

    reload_database()
    print(f"[DB] Registered: {name}")


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
            print(f"[GALLERY] Cleared {n} unknown entries. IDs reset to 001.")
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
    finally:
        is_naming = False


def embed_batch(aligned_batch):
    # aligned_batch: tensor [N, 3, 112, 112] in [0,255], RGB.
    x = (aligned_batch - 127.5) / 128.0
    x = x.to(DEVICE).half()  # Use half precision for faster inference on compatible GPUs
    with torch.no_grad():
        feats, _ = adaface(x)
    feats = feats.float()  # cast back to fp32 for DB matmul (KNOWN_EMBS is fp32)
    feats = feats / (torch.norm(feats, dim=1, keepdim=True) + 1e-8)
    return feats


class UnknownGallery:
    """RAM-only short-term memory of strangers. Same person → same Unknown_XXX ID across frames."""

    def __init__(self, match_thresh=UNKNOWN_THRESH, ttl=UNKNOWN_TTL_SECONDS, ema=UNKNOWN_EMA_ALPHA):
        self.entries = {}           # id -> {'feat': tensor[1,512], 'last_seen': epoch, 'count': int}
        self.next_id = 1
        self.match_thresh = match_thresh
        self.ttl = ttl
        self.ema = ema
        self.lock = threading.Lock()

    def assign(self, feat, now):
        """Returns (id, similarity_to_existing). feat: [1, 512] fp32 unit-normalized."""
        with self.lock:
            if self.entries:
                ids = list(self.entries.keys())
                stack = torch.cat([self.entries[i]['feat'] for i in ids], dim=0)
                sims = torch.mm(feat, stack.t())
                max_val, max_idx = torch.max(sims, dim=1)
                if max_val.item() >= self.match_thresh:
                    eid = ids[max_idx.item()]
                    e = self.entries[eid]
                    if self.ema > 0:
                        merged = self.ema * feat + (1 - self.ema) * e['feat']
                        merged = merged / (torch.norm(merged, dim=1, keepdim=True) + 1e-8)
                        e['feat'] = merged
                    e['last_seen'] = now
                    e['count'] += 1
                    return eid, float(max_val.item())
            eid = self.next_id
            self.next_id += 1
            self.entries[eid] = {'feat': feat.detach().clone(), 'last_seen': now, 'count': 1}
            return eid, 0.0

    def evict(self, now):
        with self.lock:
            stale = [k for k, v in self.entries.items() if now - v['last_seen'] > self.ttl]
            for k in stale:
                del self.entries[k]

    def get_feat(self, eid):
        with self.lock:
            e = self.entries.get(eid)
            return e['feat'].clone() if e else None

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
            return {k: {'count': v['count'], 'last_seen': v['last_seen']} for k, v in self.entries.items()}


unknown_gallery = UnknownGallery()


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


def process_frame(frame, now):
    """Run detect+embed+match on a single frame. Returns new_results list."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    with torch.no_grad():
        dets = detector.detect_faces(frame, conf_threshold=DET_THRESH, nms_threshold=NMS_THRESH)

    new_results = []

    if dets is not None and len(dets) > 0:
        crops, kept_boxes = [], []
        for det in dets:
            box = det[0:4].astype(int)
            lmks = det[5:15].reshape(5, 2).astype(np.float32)

            if min(box[2]-box[0], box[3]-box[1]) < MIN_FACE_SIZE:
                continue

            crops.append(norm_crop(rgb, lmks, size=112))
            kept_boxes.append(box)

        if crops:
            batch_np = np.stack(crops, axis=0)
            batch = torch.from_numpy(batch_np).permute(0, 3, 1, 2).contiguous().to(DEVICE).half()
            feats = embed_batch(batch)

            for i, box in enumerate(kept_boxes):
                name = "Unknown"
                color = (0, 0, 255)
                feat = feats[i:i+1]

                matched_known = False
                if KNOWN_EMBS is not None and len(KNOWN_NAMES) > 0:
                    sims = torch.mm(feat, KNOWN_EMBS.t())
                    max_val, max_idx = torch.max(sims, dim=1)
                    if max_val.item() >= THRESHOLD:
                        name = f"{KNOWN_NAMES[max_idx.item()]} ({max_val.item():.2f})"
                        color = (0, 255, 0)
                        matched_known = True

                if not matched_known:
                    eid, _ = unknown_gallery.assign(feat, now)
                    name = f"Unknown_{eid:03d}"
                    color = (0, 0, 255)

                new_results.append((box, name, color))

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
frame_counts = {vs.label: 0 for vs in active_streams}
last_results_map = {vs.label: [] for vs in active_streams}
last_evict = 0.0

print(f"[SYSTEM] Loop starting with {len(active_streams)} cam(s) at target 20+ FPS...")

while True:
    now = time.time()
    if now - last_evict > 5.0:
        unknown_gallery.evict(now)
        last_evict = now

    for vs in active_streams:
        frame = vs.read()
        if frame is None:
            continue

        frame_counts[vs.label] += 1

        if frame_counts[vs.label] % SKIP_FRAMES == 0:
            last_results_map[vs.label] = process_frame(frame, now)

        # Draw every frame for smoothness.
        for box, name, color in last_results_map[vs.label]:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(frame, name, (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow(f"VMS - {vs.label}", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('e') and not is_naming:
        is_naming = True
        threading.Thread(target=admin_input_thread, daemon=True).start()
    elif key == ord('c'):
        n = unknown_gallery.clear()
        print(f"[GALLERY] Cleared {n} unknown entries. IDs reset to 001.")
    elif key == ord('q'):
        break

for vs in streams:
    vs.stop()
cv2.destroyAllWindows()