"""checkpoint 读写工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(payload: dict[str, Any], path: str | Path) -> None:
    """保存 checkpoint 字典。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """加载 checkpoint 字典。"""
    return torch.load(Path(path), map_location=map_location)

