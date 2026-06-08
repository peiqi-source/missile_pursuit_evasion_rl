"""追逃环境奖励函数。"""

from __future__ import annotations

import numpy as np


def compute_end_to_end_sac_reward(
    action: np.ndarray | float,
    distance_trace: list[float],
    red_state: np.ndarray,
    blue_state: np.ndarray,
    dqz: float,
    kill_radius: float = 7.0,
) -> tuple[float, dict[str, float]]:
    """计算当前端到端 SAC 环境奖励。

    参数:
        action: 智能体输出的红方横向机动控制量。
        distance_trace: 当前 episode 已记录的相对距离序列。
        red_state: 红方飞行器状态向量。
        blue_state: 蓝方拦截弹状态向量。
        dqz: 方位视线角速度。
        kill_radius: 拦截判定半径。

    返回:
        reward: 标量奖励。
        reward_info: 各奖励项明细，便于训练后分析。
    """
    action_value = float(np.asarray(action).reshape(-1)[0])
    min_distance = min(distance_trace) if distance_trace else float("inf")

    # action_energy_penalty：动作能量惩罚，抑制红方过大横向过载。
    action_energy_penalty = -0.015 * action_value**2

    passed_interceptor = float(blue_state[0] - red_state[0]) <= 0.05
    if np.isfinite(min_distance) and passed_interceptor and min_distance >= kill_radius:
        # min_distance_bonus：成功脱靶后按最小距离给额外奖励。
        min_distance_bonus = 160.0 + 20.0 * min_distance
    elif np.isfinite(min_distance) and passed_interceptor and min_distance < kill_radius:
        # miss_distance_term：被近距拦截时仍保留最小距离引导信号。
        min_distance_bonus = 20.0 * min_distance
    else:
        min_distance_bonus = 0.0

    # los_rate_reward：视线角速度奖励，鼓励制造横向几何变化。
    los_rate_reward = 0.15 * 0.7 * np.log(abs(float(dqz)) + 1e-9)

    lateral_miss_distance = abs(float(red_state[2] - blue_state[2]))
    if lateral_miss_distance > kill_radius:
        # lateral_separation_reward：横向脱靶距离奖励。
        lateral_separation_reward = (lateral_miss_distance - kill_radius) * 0.005
    else:
        # lateral_separation_reward：横向距离不足时给固定惩罚。
        lateral_separation_reward = -0.15

    reward = (
        action_energy_penalty
        + min_distance_bonus
        + los_rate_reward
        + lateral_separation_reward
    )
    reward_info = {
        "action_energy_penalty": float(action_energy_penalty),
        "min_distance_bonus": float(min_distance_bonus),
        "los_rate_reward": float(los_rate_reward),
        "lateral_separation_reward": float(lateral_separation_reward),
        "min_distance": float(min_distance) if np.isfinite(min_distance) else 0.0,
        "lateral_miss_distance": float(lateral_miss_distance),
    }
    return float(reward), reward_info

