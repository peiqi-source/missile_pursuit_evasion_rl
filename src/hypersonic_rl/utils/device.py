"""设备选择工具。"""

from __future__ import annotations

import torch


def get_device(prefer_cuda: bool = True) -> torch.device:
    """自动选择 CPU 或 CUDA。"""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

