"""Interactive admin input thread — keyboard 'e' opens stdin prompt to register
or wipe unknown gallery entries. Runs in a daemon thread so the main loop
keeps drawing while the user types.
"""

from app.core import state as app_state
from .gallery_known import save_to_db
from .gallery_unknown import (
    unknown_gallery,
    delete_unknown_file,
    wipe_unknown_dir,
    invalidate_eid_bindings,
)
from .gallery_body import (
    body_gallery,
    delete_body_file,
    wipe_body_dir,
)


def admin_input_thread():
    try:
        snap = unknown_gallery.snapshot()
        if not snap:
            print("\n[REGISTER] No unknowns in gallery. Nothing to register.")
            return
        visible = ", ".join(f"#{eid:03d}(seen={info['count']})" for eid, info in sorted(snap.items()))
        print(f"\n[REGISTER] Active unknowns: {visible}")
        print("[REGISTER] Type ID to register, 'clear' to wipe gallery, 'cancel' to abort.")
        raw = input(">>> ENTER UNKNOWN ID: ").strip(" \t\n\r​﻿ ")
        if raw.lower() in ('cancel', 'c', 'q', ''):
            print("[REGISTER] Cancelled.")
            return
        if raw.lower() == 'clear':
            n = unknown_gallery.clear()
            body_gallery.clear()
            wipe_unknown_dir()
            wipe_body_dir()
            for cs in app_state.cam_states.values():
                cs.reset_all_tracks()
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
        delete_body_file(eid)
        invalidate_eid_bindings(eid)
        print(f"[REGISTER] Invalidated track bindings for eid #{eid:03d}. Box flips to green within ~200ms.")
    finally:
        app_state.is_naming = False
