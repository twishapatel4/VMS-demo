"""Init-time logging — banner prints for device + GPU."""

import torch
from .config import DEVICE


def print_device_banner():
    print(f"[SYSTEM] PyTorch Device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"[SYSTEM] GPU: {torch.cuda.get_device_name(0)}")
