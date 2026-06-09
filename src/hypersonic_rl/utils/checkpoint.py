"""
checkpoint.py

作用：
    统一管理模型检查点保存和加载。

说明：
    SAC 训练时通常需要保存：
    1. Actor 网络参数；
    2. Critic 网络参数；
    3. Target Critic 网络参数；
    4. 优化器参数；
    5. 当前 episode；
    6. 当前训练步数；
    7. 训练配置。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch

from hypersonic_rl.utils.config import ensure_dir


def save_checkpoint(
    checkpoint: Dict[str, Any],
    checkpoint_dir: str | Path,
    filename: str,
    also_save_latest: bool = True,
) -> Path:
    """
    保存模型检查点。

    参数：
        checkpoint：
            需要保存的检查点字典。

        checkpoint_dir：
            检查点保存目录。

        filename：
            检查点文件名，例如 episode_100.pth。

        also_save_latest：
            是否额外保存一份 latest.pth。
            这样测试时可以直接加载最新模型。

    返回：
        checkpoint_path：
            保存后的检查点路径。
    """
    # checkpoint_directory：检查点保存目录。
    checkpoint_directory = ensure_dir(checkpoint_dir)

    # checkpoint_path：当前检查点完整保存路径。
    checkpoint_path = checkpoint_directory / filename

    torch.save(checkpoint, checkpoint_path)

    if also_save_latest:
        # latest_path：最新检查点路径。
        latest_path = checkpoint_directory / "latest.pth"
        torch.save(checkpoint, latest_path)

    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str | Path,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    加载模型检查点。

    参数：
        checkpoint_path：
            检查点文件路径。

        device：
            加载到哪个设备。
            如果为 None，则按 PyTorch 默认方式加载。

    返回：
        checkpoint：
            检查点字典。
    """
    # path：检查点路径对象。
    path = Path(checkpoint_path)

    if not path.exists():
        raise FileNotFoundError(f"检查点文件不存在：{path}")

    # map_location：PyTorch 加载设备映射。
    map_location = device if device is not None else "cpu"

    checkpoint = torch.load(path, map_location=map_location)

    if not isinstance(checkpoint, dict):
        raise ValueError(f"检查点内容必须是字典：{path}")

    return checkpoint


def build_checkpoint(
    episode: int,
    global_step: int,
    actor_state_dict: Dict[str, Any],
    critic_state_dict: Dict[str, Any],
    target_critic_state_dict: Dict[str, Any],
    actor_optimizer_state_dict: Dict[str, Any],
    critic_optimizer_state_dict: Dict[str, Any],
    extra_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    构造标准 SAC 检查点字典。

    参数：
        episode：
            当前训练回合编号。

        global_step：
            当前全局环境交互步数。

        actor_state_dict：
            Actor 网络参数。

        critic_state_dict：
            Critic 网络参数。

        target_critic_state_dict：
            Target Critic 网络参数。

        actor_optimizer_state_dict：
            Actor 优化器参数。

        critic_optimizer_state_dict：
            Critic 优化器参数。

        extra_info：
            额外信息，例如配置、熵系数、训练曲线等。

    返回：
        checkpoint：
            标准检查点字典。
    """
    # checkpoint：统一格式的检查点字典。
    checkpoint = {
        "episode": int(episode),
        "global_step": int(global_step),
        "actor": actor_state_dict,
        "critic": critic_state_dict,
        "target_critic": target_critic_state_dict,
        "actor_optimizer": actor_optimizer_state_dict,
        "critic_optimizer": critic_optimizer_state_dict,
    }

    if extra_info is not None:
        checkpoint["extra_info"] = extra_info

    return checkpoint