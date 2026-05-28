"""GPU placement audit — prints device/dtype/param-count for each loaded model
at startup so misplaced tensors get caught before the main loop."""

import torch

from app.core.config import DEVICE
from app.core import state as app_state


def verify_gpu_placement(detector, adaface, osnet):
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

    if app_state.KNOWN_EMBS is not None:
        print(f"[INIT]   KNOWN_EMBS   device={app_state.KNOWN_EMBS.device}  dtype={app_state.KNOWN_EMBS.dtype}  N={app_state.KNOWN_EMBS.shape[0]}")
    else:
        print(f"[INIT]   KNOWN_EMBS   None (empty DB)")

    if DEVICE.type == 'cuda':
        alloc = torch.cuda.memory_allocated() / 1e9
        resv  = torch.cuda.memory_reserved()  / 1e9
        print(f"[INIT]   GPU mem      allocated={alloc:.2f}GB  reserved={resv:.2f}GB")
    print("[INIT] =============================================")
