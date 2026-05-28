"""VMS entry point.

Loads models, opens streams, runs the main draw loop, handles admin keys.
All inference logic lives in app.services.recognition.
"""

import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import torch

# --- SPEED OPTIMIZATION ---
torch.backends.cudnn.benchmark = True

from app.core.config import (
    DEVICE,
    SKIP_FRAMES,
    STATS_EVERY_FRAMES,
    CAM_SOURCES,
    ADAFACE_CHECKPOINT,
    RETINEFACE_CHECKPOINT,
)
from app.core import state as app_state
from app.core.logging import print_device_banner
from app.models.adaface import load_adaface
from app.models.retinaface import load_retinaface
from app.models.osnet import load_osnet
from app.models.gpu import verify_gpu_placement
from app.services.gallery_known import reload_database
from app.services.gallery_unknown import (
    unknown_gallery,
    load_unknowns_from_disk,
    persist_all_unknowns,
    wipe_unknown_dir,
)
from app.services.gallery_body import (
    body_gallery,
    load_bodies_from_disk,
    persist_all_bodies,
    wipe_body_dir,
)
from app.services.stream import VideoStream
from app.services.recognition import process_frame
from app.services.admin import admin_input_thread


def _init_stream(args):
    """Spawn a VideoStream — runs in worker thread so HTTP-cam handshakes overlap."""
    label, src = args
    t0 = time.time()
    vs = VideoStream(src=src, label=label, width=1280, height=720).start()
    print(f"[SYSTEM] {label} init in {time.time()-t0:.2f}s (opened={vs.opened})")
    return vs


def main():
    print_device_banner()

    # Load known-face DB before any GPU placement check (so the audit can report N).
    reload_database()

    # --- LOAD MODELS ---
    # detector = load_retinaface(RETINEFACE_CHECKPOINT, network='mobile0.25', half=False)
    detector = load_retinaface(RETINEFACE_CHECKPOINT, network='resnet50', half=False)
    adaface = load_adaface(ADAFACE_CHECKPOINT)
    osnet = load_osnet()
    verify_gpu_placement(detector, adaface, osnet)

    # Rehydrate unknown / body galleries from disk.
    load_unknowns_from_disk()
    load_bodies_from_disk()

    # --- STREAMS ---
    with ThreadPoolExecutor(max_workers=max(1, len(CAM_SOURCES))) as ex:
        streams = list(ex.map(_init_stream, CAM_SOURCES))

    active_streams = [vs for vs in streams if vs.opened]
    if not active_streams:
        print("[FATAL] No cameras opened. Exiting.")
        raise SystemExit(1)

    for vs in active_streams:
        w = vs.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = vs.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"[SYSTEM] {vs.label} resolution: {w}x{h}")

    # Per-stream state. Shared unknown_gallery → same person cross-cam = same Unknown_XXX.
    app_state.cam_states = {vs.label: app_state.CamState(vs.label) for vs in active_streams}
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
                last_results_map[vs.label] = process_frame(
                    frame, now,
                    app_state.cam_states[vs.label],
                    frame_counts[vs.label],
                    detector=detector, adaface=adaface, osnet=osnet,
                )

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
        if key == ord('e') and not app_state.is_naming:
            app_state.is_naming = True
            threading.Thread(target=admin_input_thread, daemon=True).start()
        elif key == ord('c'):
            n = unknown_gallery.clear()
            body_gallery.clear()
            wipe_unknown_dir()
            wipe_body_dir()
            for cs in app_state.cam_states.values():
                cs.reset_all_tracks()
            print(f"[GALLERY] Cleared {n} unknown entries (RAM + disk). IDs reset to 001.")
        elif key == ord('q'):
            persist_all_unknowns()
            persist_all_bodies()
            break

    for vs in streams:
        vs.stop()


if __name__ == "__main__":
    main()
