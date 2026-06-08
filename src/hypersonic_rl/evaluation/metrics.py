"""评估指标函数。"""

from __future__ import annotations

import numpy as np


def compute_min_distance(trace: dict[str, np.ndarray]) -> float:
    """计算单个 episode 的最小相对距离。"""
    distances = np.asarray(trace.get("distances", []), dtype=np.float64)
    if distances.size == 0:
        return float("inf")
    return float(np.min(distances))


def compute_success(min_distance: float, kill_radius: float) -> bool:
    """根据最小距离判断红方是否突防成功。"""
    return bool(min_distance >= kill_radius)


def compute_control_energy(trace: dict[str, np.ndarray]) -> float:
    """计算红方控制能量，使用动作平方和近似。"""
    actions = np.asarray(trace.get("actions", []), dtype=np.float64)
    if actions.size == 0:
        return 0.0
    return float(np.sum(actions**2))


def compute_episode_summary(
    trace: dict[str, np.ndarray], reward_total: float, kill_radius: float = 7.0
) -> dict[str, float]:
    """汇总单个 episode 奖励、最小距离、成功标志和控制能量。"""
    min_distance = compute_min_distance(trace)
    return {
        "episode_reward": float(reward_total),
        "min_distance": float(min_distance),
        "success": float(compute_success(min_distance, kill_radius)),
        "control_energy": compute_control_energy(trace),
    }

