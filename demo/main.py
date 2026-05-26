import os
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
MIN_FACE_SIZE = 10          # px, reject tiny faces
SKIP_FRAMES = 2              # for speed: only run detection/recognition every N frames
DETECTION_SCALE=1.2

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
target_vector = None


def admin_input_thread():
    global is_naming, target_vector
    name = input("\n>>> ENTER NAME: ").strip()
    if name and target_vector is not None:
        save_to_db(name, target_vector)
    is_naming = False
    target_vector = None


def embed_batch(aligned_batch):
    # aligned_batch: tensor [N, 3, 112, 112] in [0,255], RGB.
    x = (aligned_batch - 127.5) / 128.0
    x = x.to(DEVICE).half()  # Use half precision for faster inference on compatible GPUs
    with torch.no_grad():
        feats, _ = adaface(x)
    feats = feats.float()  # cast back to fp32 for DB matmul (KNOWN_EMBS is fp32)
    feats = feats / (torch.norm(feats, dim=1, keepdim=True) + 1e-8)
    return feats

class VideoStream:
    def __init__(self, src=0, width=1280, height=720):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.ret, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
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
        self.cap.release()


# --- MAIN LOOP ---
vs = VideoStream(src=0, width=1280, height=720).start()
actual_w = vs.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_h = vs.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"[SYSTEM] Camera Resolution: {actual_w}x{actual_h}")
frame_count = 0
last_results = []  # Stores (box, name, color, feat) for persistent drawing
current_frame_unknown_feat = None

print("[SYSTEM] Loop starting at target 20+ FPS...")

while True:
    frame = vs.read()
    if frame is None:
        continue

    frame_count += 1

    # ONLY RUN AI EVERY 'SKIP_FRAMES'
    if frame_count % SKIP_FRAMES == 0:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        with torch.no_grad():
            # Detect using MobileNet (fast)
            dets = detector.detect_faces(frame, conf_threshold=DET_THRESH, nms_threshold=NMS_THRESH)
        
        # Reset results for this AI pass
        new_results = []
        current_frame_unknown_feat = None 

        if dets is not None and len(dets) > 0:
            crops, kept_boxes = [], []
            for det in dets:
                box = det[0:4].astype(int)
                lmks = det[5:15].reshape(5, 2).astype(np.float32)

                # Filter tiny faces
                if min(box[2]-box[0], box[3]-box[1]) < MIN_FACE_SIZE:
                    continue

                crops.append(norm_crop(rgb, lmks, size=112))
                kept_boxes.append(box)

            if crops:
                # Prepare batch for AdaFace (Half Precision)
                batch_np = np.stack(crops, axis=0)
                batch = torch.from_numpy(batch_np).permute(0, 3, 1, 2).contiguous().to(DEVICE).half()
                feats = embed_batch(batch)

                for i, box in enumerate(kept_boxes):
                    name = "Unknown"
                    color = (0, 0, 255)
                    feat = feats[i:i+1]

                    if KNOWN_EMBS is not None and len(KNOWN_NAMES) > 0:
                        # Vector matching
                        sims = torch.mm(feat, KNOWN_EMBS.t())
                        max_val, max_idx = torch.max(sims, dim=1)
                        
                        if max_val.item() >= THRESHOLD:
                            name = f"{KNOWN_NAMES[max_idx.item()]} ({max_val.item():.2f})"
                            color = (0, 255, 0)

                    if name == "Unknown":
                        current_frame_unknown_feat = feat
                    
                    new_results.append((box, name, color))
        
        # Update the persistent results
        last_results = new_results

    # DRAWING SECTION (Happens every frame for maximum smoothness)
    for box, name, color in last_results:
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.putText(frame, name, (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("VMS GPU - 20FPS", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('e') and not is_naming and current_frame_unknown_feat is not None:
        target_vector = current_frame_unknown_feat
        is_naming = True
        threading.Thread(target=admin_input_thread, daemon=True).start()
    elif key == ord('q'):
        break

vs.stop() # Clean up the thread
cv2.destroyAllWindows()