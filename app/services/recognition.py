"""Recognition orchestrator — main per-frame pipeline.

Calls detection.detect_and_filter, runs ByteTrack, dispatches per-track logic
through Branch A (cached label) / Branch B (vote + cascade), then runs the
AdaFace batch + FLOW2 hybrid cascade for VOTE tags.

Cascade order (Branch B, unknown path):
  body_crop valid?
    NO  → LEFT  : face DB fallback at UNKNOWN_THRESH, then create-new
    YES → RIGHT : body lookup → secondary face check
                  → no body match? Alice fix (face DB cross-check) before create-new
"""

import cv2
import numpy as np
import torch

from app.core.config import (
    DEVICE,
    IDENTIFY_INTERVAL,
    THRESHOLD,
    TRACK_DET_IOU_MIN,
    REEMBED_EVERY_FRAMES,
    TRACK_QUERY_BUFFER,
    GALLERY_WRITE_MIN_SCORE,
    GALLERY_WRITE_MIN_SIZE,
    SECONDARY_FACE_THRESH,
)
from app.core import state as app_state
from app.utils.geometry import iou_xyxy
from app.utils.landmarks import norm_crop, frontality_weight
from app.utils.crops import body_crop
from app.utils.embed import embed_batch, osnet_embed

from .detection import detect_and_filter
from .tracking import try_stitch_label
from .gallery_unknown import unknown_gallery, persist_unknown
from .gallery_body import body_gallery, persist_body


def process_frame(frame, now, cam_state, frame_idx, *, detector, adaface, osnet):
    """Detect + ByteTrack every frame. Branch A label cache always drawn.
    AdaFace embed + cascade match + new-tid bind fire only on identify frames
    (gated by IDENTIFY_INTERVAL).

    Unknown-path cascade follows FLOW2 hybrid:
      * body-crop-valid gate splits left (face-only) vs right (body + secondary face check)
      * Alice fix: on body NO MATCH, face DB cross-check at UNKNOWN_THRESH before CREATE NEW
    """
    # [FLOW2-RISK #4] identify gating
    do_identify = (frame_idx % IDENTIFY_INTERVAL == 0)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    kept_meta, tracker_arr = detect_and_filter(detector, frame)
    tracks = cam_state.tracker.update(tracker_arr)

    new_results = []
    pending_crops = []
    pending_tags = []   # ("UPDATE", tid, eid, weight) | ("VOTE", tid, track_box, det_box, weight)

    # [FLOW2-RISK #11] tid set used by stitching helper to skip currently-alive labels.
    alive_tids = {st.track_id for st in tracks}

    for st in tracks:
        tid = st.track_id
        track_box = st.tlbr.astype(int)

        # Best-IoU det for this track.
        best_i, best_iou = -1, 0.0
        for i, (b, _, _, _, _) in enumerate(kept_meta):
            v = iou_xyxy(track_box, b)
            if v > best_iou:
                best_iou = v
                best_i = i

        # ---- Branch A: tid already labeled → reuse cached label (drawn every frame).
        if tid in cam_state.track_to_label:
            name, color = cam_state.track_to_label[tid]
            new_results.append((track_box, name, color))

            # [FLOW2-RISK #11] update stitch cache so a tid losing event can be recovered.
            eid_for_cache = cam_state.track_to_eid.get(tid)
            cache_key = ('eid', eid_for_cache) if eid_for_cache is not None else ('known', name)
            cam_state.stitch_cache[cache_key] = {
                'label': name, 'color': color, 'eid': eid_for_cache,
                'box': track_box.copy(), 'frame_idx': frame_idx,
            }

            # EMA refresh only on identify frames; cadence + quality gates unchanged.
            if not do_identify:
                continue

            eid = cam_state.track_to_eid.get(tid)
            if eid is None or best_i < 0 or best_iou < TRACK_DET_IOU_MIN:
                continue

            last = cam_state.track_last_embed.get(tid, -10**9)
            if frame_idx - last < REEMBED_EVERY_FRAMES:
                continue

            det_box, lmks, score, fscore, identify_ok = kept_meta[best_i]
            # [FLOW2-RISK #9] don't EMA-refresh from a frame that can't be identified
            # (profile too extreme, hand-on-face). Keeps gallery clean during occlusion.
            if not identify_ok:
                continue
            w = det_box[2] - det_box[0]
            h = det_box[3] - det_box[1]
            if score < GALLERY_WRITE_MIN_SCORE or min(w, h) < GALLERY_WRITE_MIN_SIZE:
                continue

            weight = frontality_weight(fscore)
            pending_crops.append(norm_crop(rgb, lmks, size=112))
            pending_tags.append(("UPDATE", tid, eid, weight))
            cam_state.track_last_embed[tid] = frame_idx
            continue

        # ---- Branch B: unbound tid.

        # [FLOW2-RISK #11] try stitching first — runs every frame (no AdaFace needed).
        # If a recently-lost labeled tid has bbox IoU >= STITCH_IOU_MIN with this
        # tid's box, inherit its label + eid directly. No vote, instant rebind.
        stitched = try_stitch_label(cam_state, track_box, frame_idx, alive_tids)
        if stitched is not None:
            cam_state.track_to_label[tid] = (stitched['label'], stitched['color'])
            if stitched['eid'] is not None:
                cam_state.track_to_eid[tid] = stitched['eid']
            cam_state.track_last_embed[tid] = frame_idx
            new_results.append((track_box, stitched['label'], stitched['color']))
            continue

        # [FLOW2-RISK #4] new tid on non-identify frame: draw nothing per spec.
        if not do_identify:
            continue

        if best_i < 0 or best_iou < TRACK_DET_IOU_MIN:
            # Detection missed this track this frame → can't identify; placeholder.
            new_results.append((track_box, "...", (200, 200, 200)))
            continue

        det_box, lmks, _, fscore, identify_ok = kept_meta[best_i]
        # [FLOW2-RISK #9] tid is being tracked, but this frame isn't usable for ID.
        # Don't accumulate vote on degraded embed; show placeholder + wait for
        # better frame. ByteTrack keeps the track alive in the meantime.
        if not identify_ok:
            new_results.append((track_box, "...", (200, 200, 200)))
            continue
        weight = frontality_weight(fscore)
        pending_crops.append(norm_crop(rgb, lmks, size=112))
        # [FLOW2-RISK #1] VOTE tag = N-frame weighted-mean buffer (TRACK_QUERY_BUFFER).
        # Lowered from 6 to 3 vs pre-FLOW2 for snappier bind; still filters bad first frames.
        pending_tags.append(("VOTE", tid, track_box, det_box, weight))

    # AdaFace forward only when there is at least one crop (gated by do_identify above).
    if pending_crops:
        batch_np = np.stack(pending_crops, axis=0)
        batch = torch.from_numpy(batch_np).permute(0, 3, 1, 2).contiguous().to(DEVICE)
        feats = embed_batch(adaface, batch)

        for i, tag in enumerate(pending_tags):
            feat = feats[i:i+1]

            if tag[0] == "UPDATE":
                # Branch A periodic EMA refresh — unchanged from pre-FLOW2 behavior.
                _, tid, eid, weight = tag
                unknown_gallery.update_entry(eid, feat, now, weight=weight)
                persist_unknown(eid)
                continue

            # === [FLOW2-RISK #1] N-frame weighted-vote bind + [FLOW2-RISK #3] hybrid cascade ===
            _, tid, track_box, det_box, weight = tag

            # Accumulate (feat, weight) until buffer fills, then weighted-mean → cascade.
            buf = cam_state.track_query_buf.setdefault(tid, [])
            buf.append((feat, weight))

            if len(buf) < TRACK_QUERY_BUFFER:
                # Still collecting — show gray '...' placeholder while waiting.
                new_results.append((track_box, "...", (200, 200, 200)))
                continue

            # Weighted mean over buffered embeds. Heavier weight on frontal frames
            # (frontality_weight in [0.15, 1.0]) → bind embed leans on best-quality views.
            stack = torch.cat([f for f, _ in buf], dim=0)                        # [N, 512]
            w_vec = torch.tensor([w for _, w in buf], device=stack.device,
                                 dtype=stack.dtype).view(-1, 1)                  # [N, 1]
            feat = (stack * w_vec).sum(dim=0, keepdim=True) / (w_vec.sum() + 1e-8)
            feat = feat / (torch.norm(feat, dim=1, keepdim=True) + 1e-8)
            cam_state.track_query_buf.pop(tid, None)

            name = "Unknown"
            color = (0, 0, 255)
            assigned_eid = None
            matched_known = False

            # Known DB cosine check (unchanged).
            if app_state.KNOWN_EMBS is not None and len(app_state.KNOWN_NAMES) > 0:
                sims = torch.mm(feat, app_state.KNOWN_EMBS.t())
                max_val, max_idx = torch.max(sims, dim=1)
                if max_val.item() >= THRESHOLD:
                    name = f"{app_state.KNOWN_NAMES[max_idx.item()]} ({max_val.item():.2f})"
                    color = (0, 255, 0)
                    matched_known = True

            if not matched_known:
                # ----- FLOW2 unknown path -----
                bcrop = body_crop(frame, det_box) if osnet is not None else None
                body_valid = bcrop is not None
                body_written = False

                if not body_valid:
                    # LEFT BRANCH — face fallback at UNKNOWN_THRESH (FLOW2 0.40).
                    face_eid, _ = unknown_gallery.find_match(feat)
                    if face_eid is not None:
                        unknown_gallery.update_entry(face_eid, feat, now, weight=1.0)
                        assigned_eid = face_eid
                    else:
                        assigned_eid, _ = unknown_gallery.assign(feat, now, weight=1.0)
                else:
                    # RIGHT BRANCH — body lookup at OSNET_THRESH.
                    bfeat = osnet_embed(osnet, [bcrop])
                    body_eid, _ = body_gallery.find_match(bfeat)

                    if body_eid is not None:
                        # Secondary face check (identity pollution prevention).
                        face_stored = unknown_gallery.get_feat(body_eid)
                        sec_match = False
                        if face_stored is not None:
                            sec_sim = float(torch.mm(feat, face_stored.t()).item())
                            sec_match = (sec_sim >= SECONDARY_FACE_THRESH)

                        if sec_match:
                            # Same face — REUSE.
                            assigned_eid = body_eid
                            unknown_gallery.update_entry(assigned_eid, feat, now, weight=1.0)
                            body_gallery.update_entry(assigned_eid, bfeat, now, weight=1.0)
                        else:
                            # Diff face OR no face stored — CREATE NEW (FLOW2 literal).
                            assigned_eid, _ = unknown_gallery.assign(feat, now, weight=1.0)
                            body_gallery.update_entry(assigned_eid, bfeat, now, weight=1.0)
                    else:
                        # NO BODY MATCH — Alice fix: face DB cross-check before CREATE NEW.
                        face_eid, _ = unknown_gallery.find_match(feat)
                        if face_eid is not None:
                            assigned_eid = face_eid
                            unknown_gallery.update_entry(assigned_eid, feat, now, weight=1.0)
                            body_gallery.update_entry(assigned_eid, bfeat, now, weight=1.0)
                        else:
                            assigned_eid, _ = unknown_gallery.assign(feat, now, weight=1.0)
                            body_gallery.update_entry(assigned_eid, bfeat, now, weight=1.0)

                    body_written = True

                persist_unknown(assigned_eid)
                if body_written:
                    persist_body(assigned_eid)
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
