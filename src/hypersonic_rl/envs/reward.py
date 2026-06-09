"""
reward.py

作用：
    统一管理环境奖励函数。

当前奖励设计：
    1. 动作能耗惩罚；
    2. 距离保持奖励；
    3. 侧向脱靶趋势奖励；
    4. 终端突防成功 / 失败奖励。

说明：
    奖励函数会直接影响 SAC 训练效果。
    当前版本先保持清晰和可调，后续可以根据实验曲线继续改。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from hypersonic_rl.envs.dynamics import compute_relative_geometry


@dataclass
class RewardConfig:
    """
    奖励函数配置。

    属性：
        action_penalty_weight：
            动作能耗惩罚权重。

        distance_reward_weight：
            距离奖励权重。
            鼓励红方保持与蓝方的距离。

        lateral_reward_weight：
            侧向分离奖励权重。
            鼓励红方形成侧向脱靶量。

        los_rate_reward_weight：
            视线角速度奖励权重。
            适当鼓励改变视线几何关系。

        success_reward：
            回合结束时突防成功奖励。

        failure_penalty：
            回合结束时突防失败惩罚。

        kill_radius：
            杀伤半径。
            如果最小距离大于 kill_radius，则认为突防成功。
    """

    action_penalty_weight: float = 0.01
    distance_reward_weight: float = 0.001
    lateral_reward_weight: float = 0.001
    los_rate_reward_weight: float = 0.1
    success_reward: float = 100.0
    failure_penalty: float = -100.0
    kill_radius: float = 7.0


def calculate_pursuit_escape_reward(
    red_state: np.ndarray,
    blue_state: np.ndarray,
    red_action: float,
    min_distance: float,
    terminated: bool,
    config: RewardConfig,
) -> Tuple[float, Dict[str, float]]:
    """
    计算单步奖励。

    参数：
        red_state：
            红方当前状态。

        blue_state：
            蓝方当前状态。

        red_action：
            红方当前动作，即 SAC 输出的侧向过载指令或惯性后的实际过载。

        min_distance：
            当前回合截至目前的最小红蓝距离。

        terminated：
            当前回合是否自然终止。
            例如蓝方已经从红方后方穿过，说明一次拦截过程结束。

        config：
            奖励函数配置。

    返回：
        reward：
            当前步总奖励。

        reward_info：
            奖励分项信息，便于日志分析。
    """
    # distance：当前红蓝距离。
    # closing_speed：接近速度。
    # los_angle：视线角。
    # los_rate：视线角速度。
    distance, closing_speed, los_angle, los_rate = compute_relative_geometry(red_state, blue_state)

    # action_penalty：动作能耗惩罚。
    action_penalty = -float(config.action_penalty_weight) * float(red_action) ** 2

    # distance_reward：距离保持奖励。
    # 使用 distance / 1000 是为了把米尺度缩小，避免奖励过大。
    distance_reward = float(config.distance_reward_weight) * np.clip(distance / 1000.0, 0.0, 100.0)

    # lateral_distance：x-z 平面中的侧向间隔。
    lateral_distance = abs(float(blue_state[2] - red_state[2]))

    # lateral_reward：侧向分离奖励。
    lateral_reward = float(config.lateral_reward_weight) * np.clip(lateral_distance / 100.0, 0.0, 100.0)

    # los_rate_reward：视线角速度奖励。
    # 视线角速度变大通常意味着拦截几何关系被改变。
    los_rate_reward = float(config.los_rate_reward_weight) * abs(float(los_rate))

    # terminal_reward：终端奖励，默认 0。
    terminal_reward = 0.0

    if terminated:
        if float(min_distance) > float(config.kill_radius):
            terminal_reward = float(config.success_reward)
        else:
            terminal_reward = float(config.failure_penalty)

    # reward：总奖励。
    reward = action_penalty + distance_reward + lateral_reward + los_rate_reward + terminal_reward

    # reward_info：奖励分项信息。
    reward_info = {
        "reward": float(reward),
        "action_penalty": float(action_penalty),
        "distance_reward": float(distance_reward),
        "lateral_reward": float(lateral_reward),
        "los_rate_reward": float(los_rate_reward),
        "terminal_reward": float(terminal_reward),
        "distance": float(distance),
        "min_distance": float(min_distance),
        "closing_speed": float(closing_speed),
        "los_angle": float(los_angle),
        "los_rate": float(los_rate),
    }

    return float(reward), reward_info