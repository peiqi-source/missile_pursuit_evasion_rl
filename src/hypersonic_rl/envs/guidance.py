"""
guidance.py

作用：
    管理制导相关函数。

当前实现：
    1. 一阶惯性环节；
    2. 蓝方比例导引；
    3. 微分对策制导律接口预留。

说明：
    当前阶段重点是复现已有代码中的“红方 SAC 直接输出机动过载，
    蓝方使用比例导引追击”的基本闭环。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from hypersonic_rl.envs.dynamics import GRAVITY, compute_relative_geometry


@dataclass
class ProportionalNavigationConfig:
    """
    比例导引配置。

    属性：
        navigation_constant：
            比例导引系数 N。

        max_overload：
            蓝方最大侧向过载。

        tau：
            蓝方一阶惯性环节时间常数。

        compensation_gain：
            对红方机动的简单补偿系数。
            当前用于近似原始代码中蓝方对红方机动的响应。
    """

    navigation_constant: float = 4.0
    max_overload: float = 6.0
    tau: float = 0.5
    compensation_gain: float = 0.5


def apply_first_order_lag(
    command: float,
    previous_output: float,
    dt: float,
    tau: float,
) -> float:
    """
    一阶惯性环节。

    参数：
        command：
            当前时刻期望控制指令。

        previous_output：
            上一时刻惯性环节输出。

        dt：
            仿真步长。

        tau：
            一阶惯性时间常数。

    返回：
        current_output：
            当前时刻惯性环节输出。

    公式：
        output_dot = (command - output) / tau
        output_new = output_old + output_dot * dt
    """
    # safe_tau：防止 tau 过小导致数值异常。
    safe_tau = max(float(tau), 1e-8)

    # current_output：一阶惯性后的当前输出。
    current_output = float(previous_output) + (float(command) - float(previous_output)) * float(dt) / safe_tau

    return current_output


def compute_blue_proportional_navigation(
    red_state: np.ndarray,
    blue_state: np.ndarray,
    red_lateral_overload: float,
    previous_blue_overload: float,
    dt: float,
    config: ProportionalNavigationConfig,
) -> Tuple[float, float, Dict[str, float]]:
    """
    计算蓝方比例导引侧向过载指令。

    参数：
        red_state：
            红方状态向量。

        blue_state：
            蓝方状态向量。

        red_lateral_overload：
            红方当前实际侧向过载输出。
            用于给蓝方导引加入简单补偿项。

        previous_blue_overload：
            蓝方上一时刻一阶惯性后的侧向过载输出。

        dt：
            仿真步长。

        config：
            比例导引配置。

    返回：
        command_overload：
            未经过一阶惯性之前的蓝方侧向过载指令。

        inertial_overload：
            经过一阶惯性之后的蓝方实际侧向过载。

        info：
            诊断信息字典，包括距离、接近速度、视线角速度等。
    """
    # distance：红蓝距离。
    # closing_speed：接近速度。
    # los_angle：视线角。
    # los_rate：视线角速度。
    distance, closing_speed, los_angle, los_rate = compute_relative_geometry(red_state, blue_state)

    # raw_command：比例导引基础指令。
    # 这里沿用工程简化形式：nzc = -N * Vc * q_dot / g。
    raw_command = -float(config.navigation_constant) * closing_speed * los_rate / GRAVITY

    # compensation：红方机动补偿项。
    compensation = -float(config.compensation_gain) * float(red_lateral_overload)

    # command_overload：加入补偿后的蓝方指令。
    command_overload = raw_command + compensation

    # command_overload：限制在蓝方最大过载范围内。
    command_overload = float(np.clip(command_overload, -config.max_overload, config.max_overload))

    # inertial_overload：经过一阶惯性环节后的蓝方实际输出。
    inertial_overload = apply_first_order_lag(
        command=command_overload,
        previous_output=previous_blue_overload,
        dt=dt,
        tau=config.tau,
    )

    # inertial_overload：再次裁剪，避免数值超界。
    inertial_overload = float(np.clip(inertial_overload, -config.max_overload, config.max_overload))

    # info：蓝方导引诊断信息。
    info = {
        "distance": float(distance),
        "closing_speed": float(closing_speed),
        "los_angle": float(los_angle),
        "los_rate": float(los_rate),
        "blue_raw_command": float(raw_command),
        "blue_command_overload": float(command_overload),
        "blue_inertial_overload": float(inertial_overload),
    }

    return command_overload, inertial_overload, info


def differential_game_guidance_placeholder(*args, **kwargs):
    """
    微分对策制导律预留接口。

    后续用于实现论文第 4 章结构：

        攻防态势 state
            ↓
        SAC 输出微分对策制导律参数 / 导航增益
            ↓
        微分对策制导律
            ↓
        最终过载指令

    当前阶段：
        不在普通端到端 SAC 中调用。
    """
    raise NotImplementedError("微分对策制导律接口将在后续第 4 章映射阶段实现。")