Here is the updated `FLOW.md`. 

I have expanded the **Unknown Path** to explicitly show exactly what happens when the system decides to create a new entry. It now clearly shows that the **Internal ReID ID** and the **Unknown_XXX ID** are generated as a pair, and shows exactly when the Face and Body embeddings are mapped to them to prevent identity pollution.

--- START OF FILE FLOW.md ---

# Face Recognition VMS — Processing Flow

## Two In-RAM Databases

| Database        | Model          | Stores                  | Purpose                          | Shown to user?      |
|-----------------|----------------|-------------------------|----------------------------------|---------------------|
| Unknown Face DB | AdaFace IR-101 | 512-dim face embeddings | Face fallback + registration     | No (internal)       |
| ReID DB         | OSNet x0.25    | 512-dim body embeddings | Cross-camera deduplication       | No (internal ID)    |

Both DBs are keyed by the same **Internal ReID ID** (e.g., `#7`).
The **Unknown ID** (e.g., `Unknown_003`) shown on screen is mapped 1:1 to that internal ID.
*(Note: Concurrent access to these databases across multiple camera threads is protected by Thread Locks).*

---

## Main Flow

```text
  ┌─────────────────┐        ┌─────────────────┐
  │    Camera 1     │        │    Camera 2     │
  │  Local Webcam   │        │   IP Stream     │
  └────────┬────────┘        └────────┬────────┘
           │  background threads      │
           │  always hold latest      │
           └─────────────┬────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │   Every 2nd frame?   │
              └───────┬──────────────┘
                      │
           ┌── NO ────┴──── YES ─────────────────┐
           │                                      │
           ▼                                      ▼
  ┌─────────────────┐             ┌───────────────────────────┐
  │  Draw previous  │             │  [1]  RetinaFace ResNet50 │
  │  labels on      │             │  Input  : frame at 50%    │
  │  current frame  │             │  Output : bounding boxes  │
  └────────▲────────┘             │           5 landmarks     │
           │                      │           confidence score │
           │                      └──────────────┬────────────┘
           │                                     │
           │                          ┌── NO ────┴──── YES ───┐
           │                          │                        │
           │                   ┌──────┴──────┐  ┌─────────────┴──────────┐
           │                   │  No faces   │  │  [2]  Quality Filters  │
           │                   │  detected   │  │  ✗  size  < 30 px      │
           │                   └─────────────┘  │  ✗  aspect outside     │
           │                                    │     0.45 – 1.4         │
           │                                    │  ✗  landmarks outside  │
           │                                    │     bounding box       │
           │                                    │  ✗  eyes below mouth   │
           │                                    │  ✗  eyes too close     │
           │                                    └─────────────┬──────────┘
           │                                                  │
           │                                                  ▼
           │                                    ┌─────────────────────────┐
           │                                    │  [3]  Face Alignment    │
           │                                    │  5-point landmark warp  │
           │                                    │  → 112 × 112 crop       │
           │                                    │  ArcFace template       │
           │                                    └─────────────┬───────────┘
           │                                                  │
           │                                                  ▼
           │                                    ┌─────────────────────────┐
           │                                    │  [4]  AdaFace IR-101    │
           │                                    │  Input  : 112×112 face  │
           │                                    │  Output : 512-dim embed │
           │                                    │  Precision : FP16 GPU   │
           │                                    │  All faces in one batch │
           │                                    └─────────────┬───────────┘
           │                                                  │
           │                                    ┌─────────────┴───────────┐
           │                                    │  Known DB  sim ≥ 0.35?  │
           │                                    └────┬──────────────┬──────┘
           │                                        YES             NO
           │                                         │               │
           │                            ┌────────────┘               │
           │                            ▼                            │
           │                 ┌───────────────────────┐               │
           │                 │   ✅  FAST PATH         │               │
           │                 │   Identity confirmed   │               │
           │                 │   Show name in GREEN   │               │
           │                 │   OSNet is NOT called  │               │
           │                 └────────────┬───────────┘               │
           │                              │                            │
           │                              │           ┌───────────────┘
           │                              │           │
           │                              │           ▼
           │                              │    UNKNOWN PATH
           │                              │    (see below)
           │                              │           │
           └──────────────────────────────┴───────────┘
                                          │
                                          ▼
                             ┌────────────────────────┐
                             │   Draw boxes + labels  │
                             │   cv2.imshow           │
                             │   one window per cam   │
                             └────────────────────────┘
```

---

## Unknown Path

> Runs only when AdaFace does NOT match the Known DB.
> A "New Record" explicitly creates both the Internal ReID ID and the Display Unknown ID simultaneously.

```text
          Not matched in Known DB
                    │
                    ▼
       ┌────────────────────────────┐
       │  [5]  OSNet x0.25          │
       │  Expand face bounding box  │
       │  → 1.5 × wider             │
       │  → 3.5 × taller downward   │
       │  Output : 512-dim ReID emb │
       └─────────────┬──────────────┘
                     │
          Body crop valid?
                     │
         ┌── NO ─────┴──── YES ──────────────────────────┐
         │                                               │
         ▼                                               ▼
┌────────────────────┐                     ┌─────────────────────────┐
│ ACQUIRE THREAD LOCK│                     │ ACQUIRE THREAD LOCK 🔒  │
│ [6a] Face Fallback │                     │ [6b] ReID DB lookup     │
│ Unknown Face DB    │                     │ Shared across cameras   │
│ sim ≥ 0.40         │                     │ sim ≥ 0.60              │
└─────────┬──────────┘                     └──────────┬──────────────┘
          │                                           │
     ┌────┴─────┐                              ┌──────┴──────┐
    YES         NO                          MATCH         NO MATCH
     │           │                             │               │
     │           │                             ▼               │
     │           │                  ┌──────────────────────┐   │
     │           │                  │ Secondary Face Check │   │
     │           │                  │ (Identity Pollution  │   │
     │           │                  │ Prevention)          │   │
     │           │                  │ AdaFace sim ≥ 0.35?  │   │
     │           │                  └────┬────────────┬────┘   │
     │           │                      YES           NO       │
     │           │                 (Same Face)   (Diff Face)   │
     │           │                       │            │        │
     ▼           ▼                       ▼            ▼        ▼
┌─────────┐ ┌──────────────────┐    ┌─────────┐  ┌──────────────────┐
│  REUSE  │ │ CREATE NEW RECORD│    │  REUSE  │  │ CREATE NEW RECORD│
│  RECORD │ │ 1. Generate new  │    │  RECORD │  │ 1. Generate new  │
└────┬────┘ │    Internal ID   │    └────┬────┘  │    Internal ID   │
     │      │ 2. Generate new  │         │       │ 2. Generate new  │
     │      │    Unknown ID    │         │       │    Unknown ID    │
     │      │ 3. Map AdaFace   │         │       │ 3. Map AdaFace   │
     │      │    embedding     │         │       │    embedding     │
     │      └────────┬─────────┘         │       │ 4. Map OSNet     │
     │               │                   │       │    embedding     │
     └───────┬───────┘                   │       └────────┬─────────┘
             │                           │                │
             └───────────────────────────┴────────────────┘
                                         │
                                         ▼
                           ┌────────────────────────────┐
                           │ Update In-RAM tracker      │
                           │ RELEASE THREAD LOCK 🔓     │
                           └─────────────┬──────────────┘
                                         │
                                         ▼
                           ┌────────────────────────────┐
                           │ Show  Unknown_XXX  in RED  │
                           └────────────────────────────┘
```

---

## Registration (E key)

```text
  Press E
     │
     ▼
  List active unknowns (Using the mapped Unknown IDs):
  #001(seen=12)  #003(seen=5)  #007(seen=31)
     │
     ▼
  Operator types ID  →  e.g.  3
     │
     ▼
  System finds internal mapping for Unknown_003
  Takes ALL mapped face embeddings from Unknown Face DB
  Calculates the Mean (Average) of those embeddings
     │
     ▼
  Operator types name  →  e.g.  "John"
     │
     ▼
  Saved to  vms_embeddings.npy  +  vms_names.txt
  Internal ReID ID & Unknown ID are deleted from tracker 
  (Thread Lock required here)
     │
     ▼
  Next appearance → matched at Known DB step → shown as  John
```

---

## Keys

| Key | Action                                        |
|-----|-----------------------------------------------|
| `E` | Register an unknown person by ID              |
| `C` | Clear all galleries, reset IDs to 001         |
| `Q` | Quit                                          |

--- END OF FILE FLOW.md ---