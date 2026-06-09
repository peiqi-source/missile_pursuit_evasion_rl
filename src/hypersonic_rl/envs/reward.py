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
from typing import Dict, Optional, Tuple

import numpy as np

from hypersonic_rl.envs.dynamics import compute_relative_geometry


@dataclass
class RewardConfig:
    """
    RewardConfig

    作用：
        管理奖励函数参数。

    属性：
        action_penalty_weight：
            红方动作能耗惩罚权重。
            动作越大，惩罚越大，避免 SAC 学到长期满过载。

        distance_reward_weight：
            距离奖励权重。
            当前使用距离的缩放值作为弱过程奖励。

        lateral_reward_weight：
            侧向分离奖励权重。
            鼓励红方形成一定侧向脱靶量。

        los_rate_reward_weight：
            视线角速度奖励权重。
            视线角速度变化可以反映攻防几何被改变。

        distance_delta_reward_weight：
            距离变化奖励权重。
            如果当前距离比上一时刻更大，说明红方正在拉开距离。

        success_reward：
            突防成功终端奖励。

        failure_penalty：
            被拦截终端惩罚。

        kill_radius：
            杀伤半径。
            min_distance <= kill_radius 表示被拦截。
    """

    action_penalty_weight: float = 0.01
    distance_reward_weight: float = 0.001
    lateral_reward_weight: float = 0.001
    los_rate_reward_weight: float = 0.1
    distance_delta_reward_weight: float = 0.01
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
    previous_distance: Optional[float] = None,

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
    # relative_info：使用与源程序一致的相对运动定义。
    relative_info = compute_relative_geometry(red_state, blue_state)

    # distance：当前红蓝距离。
    distance = float(relative_info["distance"])

    # closing_speed：接近速度。
    # closing_speed > 0 表示双方正在接近。
    closing_speed = float(relative_info["closing_speed"])

    # los_rate：标准二维视线角速度，仅作为 reward 诊断和弱奖励使用。
    los_rate = float(relative_info["los_rate_standard"])

    # action_penalty：动作能耗惩罚。
    # 目的是避免红方长期输出极限过载。
    action_penalty = -float(config.action_penalty_weight) * float(red_action) ** 2

    # distance_reward：距离过程奖励。
    # 距离越大越安全，但权重较小，避免掩盖终端成败。
    distance_reward = float(config.distance_reward_weight) * np.clip(distance / 1000.0, 0.0, 100.0)

    # lateral_distance：侧向距离差。
    lateral_distance = abs(float(red_state[2] - blue_state[2]))

    # lateral_reward：侧向脱靶过程奖励。
    lateral_reward = float(config.lateral_reward_weight) * np.clip(lateral_distance / 100.0, 0.0, 100.0)

    # los_rate_reward：视线角速度奖励。
    los_rate_reward = float(config.los_rate_reward_weight) * abs(los_rate)

    # distance_delta_reward：距离变化奖励。
    # 如果当前距离比上一时刻大，说明红方正在拉开距离，给正奖励；
    # 如果当前距离比上一时刻小，说明蓝方正在逼近，给负奖励。
    if previous_distance is None:
        distance_delta = 0.0
        distance_delta_reward = 0.0
    else:
        distance_delta = float(distance) - float(previous_distance)
        distance_delta_reward = float(config.distance_delta_reward_weight) * np.clip(
            distance_delta / 100.0,
            -10.0,
            10.0,
        )

    # intercepted：是否被拦截。
    # 只要历史最小距离小于杀伤半径，就认为发生拦截。
    intercepted = bool(float(min_distance) <= float(config.kill_radius))

    # terminal_reward：终端奖励。
    terminal_reward = 0.0

    if terminated:
        if intercepted:
            terminal_reward = float(config.failure_penalty)
        else:
            terminal_reward = float(config.success_reward)

    # reward：总奖励。
    reward = (
            action_penalty
            + distance_reward
            + lateral_reward
            + los_rate_reward
            + distance_delta_reward
            + terminal_reward
    )

    # reward_info：奖励分项和诊断信息。
    reward_info = {
        "reward": float(reward),
        "action_penalty": float(action_penalty),
        "distance_reward": float(distance_reward),
        "lateral_reward": float(lateral_reward),
        "los_rate_reward": float(los_rate_reward),
        "distance_delta_reward": float(distance_delta_reward),
        "terminal_reward": float(terminal_reward),
        "distance_delta": float(distance_delta),
        "distance": float(distance),
        "min_distance": float(min_distance),
        "intercepted_reward_flag": float(intercepted),
        "closing_speed": float(closing_speed),
        "los_angle": float(relative_info["los_angle"]),
        "los_rate": float(los_rate),
        "relative_dx": float(relative_info["dx"]),
        "relative_dy": float(relative_info["dy"]),
        "relative_dz": float(relative_info["dz"]),
        "relative_dxdt": float(relative_info["dxdt"]),
        "relative_dydt": float(relative_info["dydt"]),
        "relative_dzdt": float(relative_info["dzdt"]),
    }

    return float(reward), reward_info