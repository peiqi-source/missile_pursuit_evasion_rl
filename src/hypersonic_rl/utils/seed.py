"""
seed.py

作用：
    统一设置随机种子，保证实验尽可能可复现。

说明：
    强化学习实验中，随机性主要来自：
    1. Python random；
    2. NumPy；
    3. PyTorch CPU；
    4. PyTorch CUDA；
    5. 环境 reset 随机初始状态。
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

try:
    import torch
except ImportError:
    torch = None


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """
    设置全局随机种子。

    参数：
        seed：
            随机种子整数。
            相同 seed 通常能得到更接近的实验结果。

        deterministic：
            是否尽量启用 PyTorch 确定性计算。
            True 会降低部分速度，但更利于复现实验。
    """
    # seed_value：转换后的整数种子，避免外部传入 numpy int 等类型造成兼容问题。
    seed_value = int(seed)

    os.environ["PYTHONHASHSEED"] = str(seed_value)

    random.seed(seed_value)
    np.random.seed(seed_value)

    if torch is not None:
        torch.manual_seed(seed_value)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def seed_env(env: object, seed: Optional[int] = None) -> None:
    """
    尝试为 Gym 环境设置随机种子。

    参数：
        env：
            Gym 环境对象。

        seed：
            环境随机种子。
            如果为 None，则不执行任何操作。

    说明：
        不同 Gym 版本的 seed 接口略有差异，所以这里做兼容处理。
    """
    if seed is None:
        return

    # seed_value：环境随机种子。
    seed_value = int(seed)

    if hasattr(env, "reset"):
        try:
            env.reset(seed=seed_value)
        except TypeError:
            pass

    if hasattr(env, "action_space") and hasattr(env.action_space, "seed"):
        env.action_space.seed(seed_value)

    if hasattr(env, "observation_space") and hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed_value)