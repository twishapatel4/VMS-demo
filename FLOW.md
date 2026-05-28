# Face Recognition VMS — Processing Flow (NewFlow branch)

Adds ByteTrack-based per-camera tracking, two-stage detection (loose for tracker / strict for identify), frontality-weighted N-frame vote, identify-every-2nd-frame gating, and label stitching on top of the hybrid face+body cascade from FLOW2.

---

## Models

Each row lists the input/output tensor shape and the configurable argument(s) that can change it. Values in **{braces}** are the alternative options accepted by the constructor; the **bold** value is what main.py uses today.

### [1] RetinaFace — Face Detection

| Field            | Spec                                                                                                         |
|------------------|--------------------------------------------------------------------------------------------------------------|
| Constructor      | `RetinaFace(network_name, half, device)`                                                                     |
| `network_name`   | **`'resnet50'`** {`'mobile0.25'`, `'resnet50'`}                                                              |
| Input            | BGR uint8 image `[H, W, 3]`. Any H/W (FPN handles arbitrary size). `DETECT_SCALE` knob downscales before fwd |
| Internal forward | tensor `[1, 3, H, W]` float32. Train-time `image_size` cfg: **840** (re50) / 640 (mnet) — inference free     |
| FPN stages       | 3 levels (anchors per pixel = 3). Strides 8 / 16 / 32                                                        |
| `in_channel`     | **256** (re50) / 32 (mnet)                                                                                   |
| `out_channel`    | **256** (re50) / 64 (mnet)                                                                                   |
| `half`           | **`False`** {`True`, `False`} — FP16 forward toggle                                                          |
| Output (per det) | `[15]` = bbox `[x1,y1,x2,y2]` (4) + score (1) + landmarks `[5×2]` (10) — eye_r, eye_l, nose, mouth_r, mouth_l |
| Output (frame)   | `[N, 15]` filtered by `conf_threshold` + `nms_threshold`                                                     |
| Runtime knobs    | `conf_threshold` **0.30** (`DET_THRESH`), `nms_threshold` **0.25** (`NMS_THRESH`)                            |

### [2] ByteTrack — Per-Camera Motion Tracker

| Field            | Spec                                                                                                       |
|------------------|------------------------------------------------------------------------------------------------------------|
| Constructor      | `BYTETracker(track_thresh, match_thresh, track_buffer, frame_rate)`                                        |
| `track_thresh`   | **0.30** {float ∈ [0,1]} — high/low det split + min score to confirm new track                             |
| `match_thresh`   | **0.9** {float ∈ [0,1]} — IoU first-pass association cost gate                                             |
| `track_buffer`   | **180** {int frames} — lost track survival (≈ 6 s @ 30 fps)                                                |
| `frame_rate`     | **30** {int} — nominal FPS, scales effective buffer = `int(frame_rate/30 * track_buffer)`                  |
| Input (`update`) | numpy `[N, 5]` float32 = `[x1, y1, x2, y2, score]` per det                                                 |
| Output           | list of `STrack` — each exposes `track_id` (int), `tlbr` `[4]` `[x1,y1,x2,y2]`, `score`                    |
| Internal state   | Kalman 8-dim `[cx, cy, a, h, vx, vy, va, vh]` per track                                                    |

### [3] Face Alignment + AdaFace IR-101 — Face Embedding

| Field              | Spec                                                                                                       |
|--------------------|------------------------------------------------------------------------------------------------------------|
| Alignment          | `cv2.estimateAffinePartial2D(lmks[5,2] → ARCFACE_DST[5,2])` → `cv2.warpAffine` → 112×112 RGB uint8         |
| ArcFace template   | 5 fixed points in 112×112 (eye_r, eye_l, nose, mouth_r, mouth_l)                                           |
| Constructor        | `build_model(model_name)` → `IR_xxx(input_size)`                                                           |
| `model_name`       | **`'ir_101'`** {`'ir_18'`, `'ir_34'`, `'ir_50'`, `'ir_se_50'`, `'ir_101'`}                                  |
| `input_size`       | **`(112, 112)`** {`(112, 112)`, `(224, 224)`}                                                              |
| `num_layers`       | **100** {18, 34, 50, 100, 152, 200} — bound by `model_name`                                                |
| `mode`             | **`'ir'`** {`'ir'`, `'ir_se'`} — squeeze-excite on/off                                                     |
| Backbone channels  | `[64 → 64 → 128 → 256 → 512]` for ir_101; output_channel = 512 (≤100 layers) / 2048 (>100)                 |
| Input (batch)      | `[N, 3, 112, 112]` float32, normalized `(x - 127.5) / 128.0`                                               |
| Output             | `(features, norm)` — `features[N, 512]` float, L2-normalized in `embed_batch()`                            |
| Precision          | FP32 (`.half()` available — currently commented in main.py)                                                |

### [4] Body Crop + OSNet x0.25 — Body ReID

| Field              | Spec                                                                                                       |
|--------------------|------------------------------------------------------------------------------------------------------------|
| Body crop          | Face bbox expanded: width × **`scale_w=1.5`**, height down × **`scale_h=3.5`**, up by `0.2 × face_h`       |
| Crop validity gate | `w ≥ BODY_CROP_MIN_W=80`, `h ≥ BODY_CROP_MIN_H=40` (else returns `None` → routes LEFT branch)              |
| Constructor        | `osnet_xN(num_classes, pretrained, loss, **kwargs)`                                                        |
| Variant            | **`osnet_x0_25`** {`osnet_x0_25`, `osnet_x0_5`, `osnet_x0_75`, `osnet_x1_0`, `osnet_ibn_x1_0`}             |
| Backbone channels  | **`[16, 64, 96, 128]`** (x0.25) / `[32,128,192,256]` (x0.5) / `[48,192,288,384]` (x0.75) / `[64,256,384,512]` (x1.0) |
| `num_classes`      | **1000** {int} — classifier head ignored at eval (stripped in load_osnet)                                  |
| `loss`             | **`'softmax'`** {`'softmax'`, `'triplet'`}                                                                 |
| `feature_dim`      | **512** {int} — FC output, same across all width variants                                                  |
| `IN`               | **`False`** {`True`, `False`} — instance-norm flag (`osnet_ibn_*` uses `True`)                             |
| `pretrained`       | **`False`** in load_osnet (weights restored manually from checkpoint)                                      |
| Input              | `[N, 3, 256, 128]` float32 — `OSNET_INPUT_H=256`, `OSNET_INPUT_W=128`. ImageNet mean/std normalized        |
| Output             | `[N, 512]` float, L2-normalized in `osnet_embed()`                                                         |

---

### Knobs at a glance

| Module     | Argument           | Current value     | Alternatives                                                    |
|------------|--------------------|-------------------|------------------------------------------------------------------|
| RetinaFace | `network_name`     | `resnet50`        | `mobile0.25`, `resnet50`                                         |
| RetinaFace | `half`             | `False`           | `True`, `False`                                                  |
| RetinaFace | `conf_threshold`   | 0.30              | float ∈ [0,1]                                                    |
| RetinaFace | `nms_threshold`    | 0.25              | float ∈ [0,1]                                                    |
| ByteTrack  | `track_thresh`     | 0.30              | float ∈ [0,1]                                                    |
| ByteTrack  | `match_thresh`     | 0.9               | float ∈ [0,1]                                                    |
| ByteTrack  | `track_buffer`     | 180               | int frames                                                       |
| ByteTrack  | `frame_rate`       | 30                | int FPS                                                          |
| AdaFace    | `model_name`       | `ir_101`          | `ir_18`, `ir_34`, `ir_50`, `ir_se_50`, `ir_101`                  |
| AdaFace    | `input_size`       | `(112, 112)`      | `(112, 112)`, `(224, 224)`                                       |
| AdaFace    | `mode`             | `ir`              | `ir`, `ir_se`                                                    |
| OSNet      | variant            | `osnet_x0_25`     | `osnet_x0_25`, `osnet_x0_5`, `osnet_x0_75`, `osnet_x1_0`, `osnet_ibn_x1_0` |
| OSNet      | `feature_dim`      | 512               | int (FC width)                                                   |
| OSNet      | `loss`             | `softmax`         | `softmax`, `triplet`                                             |
| OSNet      | input H×W          | 256×128           | any, set via `OSNET_INPUT_H` / `OSNET_INPUT_W`                   |
| Body crop  | `scale_w`          | 1.5               | float multiplier of face width                                   |
| Body crop  | `scale_h`          | 3.5               | float multiplier of face height (downward)                       |

---

## State

### In-RAM Galleries

| Gallery           | Model          | Stores                     | Threshold | Purpose                              | Persisted to disk            |
|-------------------|----------------|----------------------------|-----------|--------------------------------------|------------------------------|
| Known DB          | AdaFace IR-101 | 512-dim face mean per name | 0.35      | Registered identities                | `embeddings/known/*.npy`     |
| Unknown Face Gal. | AdaFace IR-101 | 512-dim face embed (Top-1) | 0.40      | Face fallback + cross-frame stranger | `embeddings/unknown/*.npy`   |
| Body ReID Gal.    | OSNet x0.25    | 512-dim body embed (Top-1) | 0.80      | Cross-camera body match              | `embeddings/body/*.npy`      |

Unknown Face & Body keyed by same `eid` → display ID `Unknown_NNN`.
Top-K=1 with weighted EMA (frontality scales α).
Thread Lock guards both galleries.

### Per-Camera State (`CamState`)

| Field               | Maps                          | Purpose                                                  |
|---------------------|-------------------------------|----------------------------------------------------------|
| `tracker`           | —                             | ByteTracker (Kalman + IoU, 6s buffer)                    |
| `track_to_label`    | tid → (name, color)           | Cached label drawn every frame (Branch A)                |
| `track_to_eid`      | tid → eid                     | Only set for Unknown_NNN binds                           |
| `track_query_buf`   | tid → list[(feat, weight)]    | N-frame vote buffer for new tid                          |
| `track_last_embed`  | tid → frame_idx               | Cadence gate for EMA refresh                             |
| `stitch_cache`      | (eid\|name) → {box, frame_idx}| Recover label on tid swap within 4s + IoU≥0.3            |

---

## Main Flow

```text
  ┌─────────────────┐        ┌─────────────────┐
  │    Camera 1     │        │    Camera 2     │
  │  Local Webcam   │        │   IP Stream     │
  └────────┬────────┘        └────────┬────────┘
           │ bg threads hold latest   │
           └─────────────┬────────────┘
                         │
                         ▼
              ┌─────────────────────────┐
              │  [1] RetinaFace R50     │
              │  Every frame            │
              │  Input  : raw frame     │
              │  Output : bbox + lmks   │
              │            + det score  │
              └────────────┬────────────┘
                           │
              ┌────────────┴────────────┐
              │ Stage 1 — LOOSE gates   │
              │  ✗ size < 20 px         │
              │  ✗ aspect ∉ [0.30, 1.4] │
              │  ✗ landmarks far OOB    │
              │  ✗ eyes below mouth     │
              └────────────┬────────────┘
                           │ passes feed ByteTrack
                           ▼
              ┌─────────────────────────┐
              │  [2] ByteTrack update   │
              │  match_thresh = 0.9     │
              │  buffer = 180 (~6s)     │
              │  Output : alive tids    │
              └────────────┬────────────┘
                           │
              ┌────────────┴────────────┐
              │ Stage 2 — STRICT gates  │
              │  set identify_ok flag   │
              │  ✗ strict landmark geom │
              │  ✗ frontality < 0.20    │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │  Per alive tid:         │
              │  best-IoU det match     │
              │  (IoU ≥ 0.3 required)   │
              └────────────┬────────────┘
                           │
              ┌── tid in track_to_label ──┐
             YES                          NO
              │                            │
              ▼                            ▼
       BRANCH A (cached)            BRANCH B (unbound)
       (see below)                  (see below)
              │                            │
              └──────────────┬─────────────┘
                             ▼
                   ┌─────────────────────┐
                   │ Draw boxes + labels │
                   │ Every frame, even   │
                   │ between identify    │
                   │ frames              │
                   └─────────────────────┘
```

---

## Branch A — Cached tid (already labeled)

Runs every frame. AdaFace embed only on identify frames.

```text
  tid already in track_to_label
            │
            ▼
   Draw cached (name, color)
   Update stitch_cache[eid|name] = {box, frame_idx}
            │
            ▼
   do_identify  (frame_idx % 2 == 0) ?
            │
       ┌── NO ──── YES ──────────────────┐
       │                                  │
   skip refresh                           ▼
                            ┌─────────────────────────┐
                            │ Gates for EMA refresh:  │
                            │  ✓ track_to_eid known   │
                            │  ✓ best_iou ≥ 0.3       │
                            │  ✓ frame_idx - last     │
                            │     ≥ REEMBED (15)      │
                            │  ✓ identify_ok          │
                            │  ✓ det_score ≥ 0.90     │
                            │  ✓ face size ≥ 40 px    │
                            └────────────┬────────────┘
                                         │
                                         ▼
                            ┌─────────────────────────┐
                            │ [3a] Face Alignment     │
                            │  5-pt warp → 112×112    │
                            │ [3b] AdaFace IR-101     │
                            │  → 512-dim embed        │
                            │ weight = f(frontality)  │
                            │ unknown_gallery         │
                            │   .update_entry(eid,    │
                            │     feat, weight)       │
                            │ persist_unknown(eid)    │
                            └─────────────────────────┘
```

---

## Branch B — Unbound tid (new or label cleared)

```text
   tid NOT in track_to_label
            │
            ▼
   do_identify ?
            │
       ┌── NO ──── YES ──────────────────┐
       │                                  │
   draw nothing                           ▼
   (wait next id frame)        best-IoU det exists?
                                          │
                                ┌── NO ───┴── YES ──────┐
                                │                        │
                                ▼                        ▼
                         placeholder        identify_ok flag set?
                         "..." gray                      │
                                                ┌── NO ──┴── YES ─────┐
                                                │                      │
                                                ▼                      ▼
                                         placeholder "..."     [3a] Face Alignment
                                                                5-pt warp → 112×112
                                                               [3b] AdaFace IR-101
                                                                → 512-dim embed
                                                                       │
                                                                       ▼
                                                            ┌─────────────────────┐
                                                            │ Push (feat, weight) │
                                                            │ into vote buffer    │
                                                            │ (TRACK_QUERY_BUFFER │
                                                            │  = 3 frames)        │
                                                            └─────────┬───────────┘
                                                                      │
                                                          buf full (3 entries)?
                                                                      │
                                                              ┌── NO ─┴── YES ──┐
                                                              │                  │
                                                              ▼                  ▼
                                                       draw "..."     Weighted mean of
                                                                      buffered embeds
                                                                      → L2 norm
                                                                      → CASCADE (below)
```

### Branch B Cascade — bind label

```text
                feat (weighted-mean over 3 frames)
                          │
                          ▼
              ┌─────────────────────────┐
              │ Known DB cosine ≥ 0.35? │
              └──────┬────────────┬─────┘
                    YES           NO
                     │             │
                     ▼             │
            ┌──────────────────┐   │
            │  FAST PATH       │   │
            │  Green label     │   │
            │  "Name (0.NN)"   │   │
            │  track_to_eid    │   │
            │  NOT set         │   │
            └──────────────────┘   │
                                   ▼
                         ┌────────────────────┐
                         │ body_crop valid?   │
                         │ w≥80 & h≥40        │
                         └─────┬──────────┬───┘
                              NO         YES
                               │          │
                               ▼          ▼
                         LEFT BRANCH    RIGHT BRANCH
                         face-only      (body + secondary)
                         (see below)    (see below)
```

#### LEFT — body crop invalid (face-only)

```text
   unknown_gallery.find_match(feat) ≥ 0.40 ?
            │
       ┌── YES ──── NO ──┐
       │                  │
       ▼                  ▼
   REUSE eid         CREATE NEW eid
   EMA blend         (unknown_gallery.assign)
   feat              face embed mapped
                     body embed NOT mapped
            │
            ▼
   persist_unknown(eid)
```

#### RIGHT — body crop valid

```text
   [4] OSNet x0.25
   Expand face bbox → 1.5× wider, 3.5× taller down
   256×128 RGB crop → 512-dim ReID embed
            │
            ▼
   body_gallery.find_match(bfeat) ≥ 0.80 ?
            │
       ┌── MATCH ───── NO MATCH ────────┐
       │                                  │
       ▼                                  ▼
   Secondary face check          Alice fix:
   face_stored = ug.get_feat     unknown_gallery.find_match(feat) ≥ 0.40?
   sim(feat, face_stored) ≥ 0.35?            │
       │                              ┌── YES ──── NO ──┐
   ┌── YES ──── NO ──┐                │                  │
   │                 │                 ▼                  ▼
   ▼                 ▼            REUSE face_eid    CREATE NEW eid
 REUSE          CREATE NEW eid    map face + body   map face + body
 body_eid       map face + body   embed             embed
 EMA blend      (unknown_gallery
 face + body    .assign)
                map face + body
                embed
            │
            ▼
   persist_unknown(eid) + persist_body(eid)
   color = RED, name = Unknown_NNN
```

After cascade:
- `track_to_label[tid] = (name, color)`
- `track_to_eid[tid] = eid` (only for Unknown)
- `track_last_embed[tid] = frame_idx`

---

## Track Stitching — Recover label across tid swap

Runs as side-effect of Branch A. When ByteTrack drops a labeled tid (occlusion > 6s) and reassigns a fresh tid for the same person, the new tid can inherit the old label without re-voting.

```text
   Every Branch A draw:
     stitch_cache[('eid', eid) | ('known', name)] =
       {box, frame_idx, label, color, eid}

   New tid in Branch B (before vote starts):
     try_stitch_label(track_box, frame_idx, alive_tids)
       ↳ skip cache entries whose label is alive on another tid
       ↳ pick highest IoU ≥ 0.3 within last 120 frames (~4s)
       ↳ on hit: copy (label, color, eid) into new tid → skip vote
```

Currently `_try_stitch_label()` defined but **not yet wired into Branch B** in this branch — stitch cache populated only.

---

## Identify Gating

`IDENTIFY_INTERVAL = 2` → AdaFace embed + cascade + Branch A EMA refresh run every 2nd frame. Detection + ByteTrack run every frame so tracks stay alive. Cached labels still drawn on skipped frames for smooth motion.

`SKIP_FRAMES = 1` at top loop → `process_frame` runs every grabbed frame.

---

## Disk Persistence

Auto-load at startup:
- `load_unknowns_from_disk()` — repopulate face gallery
- `load_bodies_from_disk()` — repopulate body gallery

Per-write (after every gallery update in cascade / EMA refresh):
- `persist_unknown(eid)` → `embeddings/unknown/Unknown_NNN.npy`
- `persist_body(eid)`    → `embeddings/body/Body_NNN.npy`

On `q` exit:
- `persist_all_unknowns()` + `persist_all_bodies()`

On `c` clear + on register:
- `delete_unknown_file(eid)` + `delete_body_file(eid)` + `invalidate_eid_bindings(eid)` (drops cached track labels + stitch entries across all cams).

---

## Registration (E key)

```text
   Press E
       │
       ▼
   Snapshot active unknowns (with seen counts)
   #001(seen=12)  #003(seen=5)  #007(seen=31)
       │
       ▼
   Operator types Unknown ID
   special: 'clear' wipes both galleries + disk + all cam states
   special: 'cancel' aborts
       │
       ▼
   unknown_gallery.get_feat(eid) → L2-norm mean of Top-K embeds
       │
       ▼
   Operator types name
       │
       ▼
   save_to_db(name, feat)         → known/<name>.npy + reload_database()
   unknown_gallery.remove(eid)
   body_gallery.remove(eid)
   delete_unknown_file(eid) + delete_body_file(eid)
   invalidate_eid_bindings(eid)   → drop track_to_label/eid + stitch entry across cams
       │
       ▼
   Next appearance → matched at Known DB step → green label
```

Thread lock around galleries handled inside `UnknownGallery` methods.

---

## Keys

| Key | Action                                                                |
|-----|-----------------------------------------------------------------------|
| `E` | Register an unknown by ID (or `clear` / `cancel` at the prompt)       |
| `C` | Clear both galleries (RAM + disk) + reset all cam tracker bindings    |
| `Q` | Persist galleries to disk and quit                                    |

---

## Tuned Constants Reference

| Constant                  | Value | Role                                              |
|---------------------------|-------|---------------------------------------------------|
| `IDENTIFY_INTERVAL`       | 2     | AdaFace embed every 2nd frame                     |
| `TRACK_QUERY_BUFFER`      | 3     | Frames pooled before Branch B bind                |
| `REEMBED_EVERY_FRAMES`    | 15    | Branch A EMA refresh cadence                      |
| `BYTETRACK_MATCH_THRESH`  | 0.9   | IoU first-pass association                        |
| `BYTETRACK_BUFFER`        | 180   | Lost track survival ~6 s @ 30 fps                 |
| `STITCH_WINDOW_FRAMES`    | 120   | Stitch cache lifetime ~4 s                        |
| `STITCH_IOU_MIN`          | 0.3   | Min bbox overlap for stitch                       |
| `MIN_FRONTALITY`          | 0.20  | Hard-reject for identify_ok                       |
| `FRONTAL_REF`             | 0.55  | Frontality → full weight at/above                 |
| `FRONTAL_WEIGHT_FLOOR`    | 0.15  | Floor for embed weight                            |
| `THRESHOLD` (known)       | 0.35  | Known DB cosine                                   |
| `UNKNOWN_THRESH` (face)   | 0.40  | Face fallback / cross-frame                       |
| `OSNET_THRESH` (body)     | 0.80  | Body ReID                                         |
| `SECONDARY_FACE_THRESH`   | 0.35  | Identity-pollution guard on body match            |
| `BODY_CROP_MIN_W/H`       | 80/40 | Body crop validity gate                           |
| `TRACK_DET_IOU_MIN`       | 0.3   | Track ↔ det association in process_frame          |
| `GALLERY_WRITE_MIN_SCORE` | 0.90  | Det score required for Branch A EMA write         |
| `GALLERY_WRITE_MIN_SIZE`  | 40    | Min face size for Branch A EMA write              |
