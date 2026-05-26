import os
import cv2
import torch
import numpy as np
import threading

os.environ["QT_QPA_PLATFORM"] = "xcb"

from facexlib.detection import init_detection_model
from net import build_model

# --- SETTINGS ---
CHECKPOINT_PATH = 'adaface_ir101_ms1mv2.ckpt'
EMB_DB_PATH = 'vms_embeddings.npy'
NAME_DB_PATH = 'vms_names.txt'
THRESHOLD = 0.45
DET_THRESH = 0.50           # RetinaFace confidence
NMS_THRESH = 0.40
MIN_FACE_SIZE = 10          # px, reject tiny faces

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
        model.eval()
        return model
    except Exception as e:
        print(f"[FATAL] AdaFace Load Error: {e}")
        exit()


# RetinaFace (ResNet50) on GPU via facexlib. PyTorch, no onnxruntime.
# Weights auto-downloaded to ~/.cache/facexlib on first run.
detector = init_detection_model('retinaface_resnet50', half=False, device=DEVICE)

adaface = load_adaface(CHECKPOINT_PATH)

# ArcFace 5-point template for 112x112 alignment.
ARCFACE_DST = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014],
    [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]
], dtype=np.float32)


def norm_crop(img, landmarks, size=112):
    M, _ = cv2.estimateAffinePartial2D(landmarks, ARCFACE_DST, method=cv2.LMEDS)
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
    x = x.to(DEVICE)
    with torch.no_grad():
        feats, _ = adaface(x)
    feats = feats / (torch.norm(feats, dim=1, keepdim=True) + 1e-8)
    return feats


# --- MAIN LOOP ---
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"[SYSTEM] Camera Resolution: {actual_w}x{actual_h}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # RetinaFace (facexlib) expects BGR ndarray. AdaFace crops built from RGB.
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    with torch.no_grad():
        dets = detector.detect_faces(frame, conf_threshold=DET_THRESH, nms_threshold=NMS_THRESH)
    # dets: [N,15] = x1,y1,x2,y2,score, (lex,ley,rex,rey,nx,ny,lmx,lmy,rmx,rmy)

    current_frame_unknown_feat = None

    if dets is not None and len(dets) > 0:
        crops, kept = [], []
        for det in dets:
            score = float(det[4])
            box = det[0:4].astype(int)
            lmks = det[5:15].reshape(5, 2).astype(np.float32)

            w, h = box[2] - box[0], box[3] - box[1]
            if min(w, h) < MIN_FACE_SIZE:
                continue

            crops.append(norm_crop(rgb, lmks, size=112))
            kept.append((box, score))

        if crops:
            batch_np = np.stack(crops, axis=0)  # [N,112,112,3] RGB uint8
            batch = torch.from_numpy(batch_np).permute(0, 3, 1, 2).contiguous().float()
            feats = embed_batch(batch)

            for i, (box, score) in enumerate(kept):
                name = "Unknown"
                color = (0, 0, 255)

                feat = feats[i:i+1]
                if KNOWN_EMBS is not None and len(KNOWN_NAMES) > 0:
                    sims = torch.mm(feat, KNOWN_EMBS.t())
                    max_val, max_idx = torch.max(sims, dim=1)
                    idx = max_idx.item()

                    if max_val.item() >= THRESHOLD and idx < len(KNOWN_NAMES):
                        name = f"{KNOWN_NAMES[idx]} ({max_val.item():.2f})"
                        color = (0, 255, 0)

                if name == "Unknown":
                    current_frame_unknown_feat = feat

                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(frame, name, (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("VMS GPU", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('e') and not is_naming and current_frame_unknown_feat is not None:
        target_vector = current_frame_unknown_feat
        is_naming = True
        threading.Thread(target=admin_input_thread, daemon=True).start()
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()