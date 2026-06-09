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

from hypersonic_rl.envs.dynamics import (
    EPS,
    GRAVITY,
    build_velocity_vector,
    compute_relative_geometry,
)


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
    # relative_info：按源程序坐标定义计算相对位置和相对速度。
    relative_info = compute_relative_geometry(red_state, blue_state)

    # dx/dy/dz：从蓝方指向红方的相对位置。
    dx = relative_info["dx"]
    dy = relative_info["dy"]
    dz = relative_info["dz"]

    # dxdt/dydt/dzdt：红方相对蓝方速度。
    dxdt = relative_info["dxdt"]
    dydt = relative_info["dydt"]
    dzdt = relative_info["dzdt"]

    # dis_r：红蓝距离。
    dis_r = max(relative_info["distance"], EPS)

    # v_r：源程序中的相对速度，也就是距离变化率。
    # v_r < 0 表示双方正在接近。
    v_r = float((dx * dxdt + dy * dydt + dz * dzdt) / dis_r)

    if abs(v_r) < EPS:
        # tgo：预计剩余时间。
        tgo = 0.0

        # dqy/dqz：视线角速度。v_r 过小时无法稳定计算，置 0。
        dqy = 0.0
        dqz = 0.0
    else:
        # tgo：源程序时间到达估计。
        tgo = float(-dis_r / v_r)

        if abs(tgo) < EPS:
            dqy = 0.0
            dqz = 0.0
        else:
            # dqy：高低视线角速度，当前不用于蓝方侧向控制，但保留诊断。
            dqy = float(-dy / (v_r * tgo ** 2) - dydt / (v_r * tgo))

            # dqz：方位视线角速度，源程序用于蓝方侧向过载。
            dqz = float(dz / (v_r * tgo ** 2) + dzdt / (v_r * tgo))

    # raw_command：源程序比例导引指令。
    raw_command = (
            -float(config.navigation_constant) * v_r * dqz / GRAVITY
            - float(config.compensation_gain) * float(red_lateral_overload)
    )

    # command_overload：裁剪后的蓝方过载指令。
    command_overload = float(
        np.clip(raw_command, -config.max_overload, config.max_overload)
    )

    # inertial_overload：一阶惯性后的蓝方实际过载。
    inertial_overload = apply_first_order_lag(
        command=command_overload,
        previous_output=previous_blue_overload,
        dt=dt,
        tau=config.tau,
    )

    # inertial_overload：再次裁剪，保证不过载。
    inertial_overload = float(
        np.clip(inertial_overload, -config.max_overload, config.max_overload)
    )

    info = {
        **relative_info,
        "source_v_r": float(v_r),
        "source_tgo": float(tgo),
        "source_dqy": float(dqy),
        "source_dqz": float(dqz),
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