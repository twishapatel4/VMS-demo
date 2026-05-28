import os
import cv2
import torch
import numpy as np
import threading

os.environ["QT_QPA_PLATFORM"] = "xcb"

from facenet_pytorch import MTCNN
from net import build_model

# --- SETTINGS ---
CHECKPOINT_PATH = 'adaface_ir101_ms1mv2.ckpt'
EMB_DB_PATH = 'vms_embeddings.npy'
NAME_DB_PATH = 'vms_names.txt'
THRESHOLD = 0.45
DET_THRESH = 0.90  # MTCNN final-stage probability

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


# MTCNN on GPU. image_size=112 returns aligned crops sized for AdaFace.
# post_process=False keeps pixel range in [0,255] so AdaFace's own normalization applies.
mtcnn = MTCNN(
    image_size=112,
    margin=0,
    keep_all=True,
    post_process=False,
    thresholds=[0.6, 0.7, 0.7],
    device=DEVICE,
)

adaface = load_adaface(CHECKPOINT_PATH)


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
    # aligned_batch: tensor [N, 3, 112, 112] in [0,255] (from MTCNN post_process=False)
    x = (aligned_batch - 127.5) / 128.0
    x = x.to(DEVICE)
    with torch.no_grad():
        feats, _ = adaface(x)
    feats = feats / (torch.norm(feats, dim=1, keepdim=True) + 1e-8)
    return feats


# --- MAIN LOOP ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # MTCNN expects RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    boxes, probs = mtcnn.detect(rgb)
    aligned = mtcnn.extract(rgb, boxes, save_path=None) if boxes is not None else None

    current_frame_unknown_feat = None

    if aligned is not None and len(aligned) > 0:
        # aligned may be tensor [N,3,112,112] or list; normalize to tensor
        if isinstance(aligned, list):
            aligned = torch.stack([a for a in aligned if a is not None])
        if aligned.ndim == 3:
            aligned = aligned.unsqueeze(0)

        feats = embed_batch(aligned)

        for i, box in enumerate(boxes):
            if probs[i] is None or probs[i] < DET_THRESH:
                continue

            feat = feats[i:i+1]
            name = "Unknown"
            color = (0, 0, 255)

            if KNOWN_EMBS is not None and len(KNOWN_NAMES) > 0:
                sims = torch.mm(feat, KNOWN_EMBS.t())
                max_val, max_idx = torch.max(sims, dim=1)
                idx = max_idx.item()

                if max_val.item() >= THRESHOLD and idx < len(KNOWN_NAMES):
                    name = f"{KNOWN_NAMES[idx]} ({max_val.item():.2f})"
                    color = (0, 255, 0)

            if "Unknown" in name:
                current_frame_unknown_feat = feat

            b = box.astype(int)
            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), color, 2)
            cv2.putText(frame, name, (b[0], b[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

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
