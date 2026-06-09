"""
device.py

作用：
    统一管理 PyTorch 运行设备。
    自动判断使用 CPU 还是 CUDA GPU。
"""

from __future__ import annotations

from typing import Optional

import torch


def get_device(prefer_gpu: bool = True, gpu_id: int = 0) -> torch.device:
    """
    获取 PyTorch 运行设备。

    参数：
        prefer_gpu：
            是否优先使用 GPU。
            如果为 True 且 CUDA 可用，则返回 cuda:{gpu_id}。

        gpu_id：
            GPU 编号。
            单卡电脑通常为 0。

    返回：
        device：
            torch.device 对象。
    """
    # cuda_available：当前环境是否可以使用 CUDA。
    cuda_available = torch.cuda.is_available()

    if prefer_gpu and cuda_available:
        return torch.device(f"cuda:{gpu_id}")

    return torch.device("cpu")


def describe_device(device: Optional[torch.device] = None) -> str:
    """
    返回设备描述字符串。

    参数：
        device：
            需要描述的 torch.device。
            如果为 None，则自动调用 get_device()。

    返回：
        description：
            设备描述字符串。
    """
    # current_device：实际用于描述的设备对象。
    current_device = device if device is not None else get_device()

    if current_device.type == "cuda":
        # gpu_index：CUDA 设备编号。
        gpu_index = current_device.index if current_device.index is not None else 0

        # gpu_name：GPU 名称。
        gpu_name = torch.cuda.get_device_name(gpu_index)

        return f"CUDA 可用，当前设备：cuda:{gpu_index}，GPU 名称：{gpu_name}"

    return "CUDA 不可用或未启用，当前设备：CPU"


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """
    将 batch 字典中的张量移动到指定设备。

    参数：
        batch：
            样本字典，例如：
            {
                "state": tensor,
                "action": tensor,
                "reward": tensor
            }

        device：
            目标设备。

    返回：
        moved_batch：
            移动后的样本字典。
    """
    # moved_batch：移动设备后的新字典。
    moved_batch = {}

    for key, value in batch.items():
        if torch.is_tensor(value):
            moved_batch[key] = value.to(device)
        else:
            moved_batch[key] = value

    return moved_batch