"""
guidance.py

作用：
    管理 source_pn 等工程制导律计算。

当前约定：
    1. 红方仍由 SAC 输出 1 维横向过载动作；
    2. 拦截弹制导律输出内部 ny / nz 双通道过载；
    3. 所有对外诊断字段使用 interceptor 命名，不再暴露旧单通道字段。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from hypersonic_rl.envs.autopilot import FirstOrderAutopilot, limit_planar_overload
from hypersonic_rl.envs.dynamics import EPS, GRAVITY, compute_relative_geometry


@dataclass
class ProportionalNavigationConfig:
    """
    比例导引配置。

    属性：
        navigation_constant：
            比例导引系数 N。
        max_overload：
            拦截弹最大可用机动过载，单位 g。
        tau：
            拦截弹一阶自动驾驶仪时间常数。
        compensation_gain：
            对红方横向机动的前馈补偿系数。
    """

    navigation_constant: float = 4.0
    max_overload: float = 6.0
    tau: float = 0.5
    compensation_gain: float = 0.5


def compute_interceptor_proportional_navigation(
    red_state: np.ndarray,
    interceptor_state: np.ndarray,
    red_lateral_overload: float,
    previous_interceptor_ny_actual: float,
    previous_interceptor_nz_actual: float,
    dt: float,
    config: ProportionalNavigationConfig,
) -> Tuple[Tuple[float, float], Tuple[float, float], Dict[str, float]]:
    """
    计算 source_pn 模式下单枚拦截弹的双通道比例导引输出。

    参数：
        red_state：
            红方当前状态向量。
        interceptor_state：
            拦截弹当前状态向量。
        red_lateral_overload：
            红方经过一阶自动驾驶仪后的实际横向过载。
        previous_interceptor_ny_actual：
            拦截弹上一仿真步纵向实际过载。
        previous_interceptor_nz_actual：
            拦截弹上一仿真步侧向实际过载。
        dt：
            仿真步长，单位 s。
        config：
            source_pn 比例导引配置。

    返回：
        command_pair：
            拦截弹过载指令二元组，格式为 (ny_command, nz_command)。
        actual_pair：
            拦截弹一阶自动驾驶仪实际输出二元组，格式为 (ny_actual, nz_actual)。
        info：
            制导诊断字段。
    """
    # relative_info：从拦截弹指向红方的相对几何。
    relative_info = compute_relative_geometry(red_state, interceptor_state)

    # dy/dz：纵向和侧向相对位置分量。
    dy = float(relative_info["dy"])
    dz = float(relative_info["dz"])

    # dydt/dzdt：纵向和侧向相对速度分量。
    dydt = float(relative_info["dydt"])
    dzdt = float(relative_info["dzdt"])

    # distance：三维相对距离，设置下限避免除零。
    distance = max(float(relative_info["distance"]), EPS)

    # range_rate：距离变化率，小于 0 表示双方正在接近。
    range_rate = float(relative_info["range_rate"])

    if abs(range_rate) < EPS:
        # tgo/dqy/dqz：距离变化率过小时不计算剩余飞行时间和修正视线角速度。
        tgo = 0.0
        dqy = 0.0
        dqz = 0.0
    else:
        # tgo：沿用原始公式的预计交会时间。
        tgo = float(-distance / range_rate)

        if abs(tgo) < EPS:
            # dqy/dqz：tgo 过小时避免数值爆炸。
            dqy = 0.0
            dqz = 0.0
        else:
            # dqy：纵向修正视线角速度。
            dqy = float(-dy / (range_rate * tgo**2) - dydt / (range_rate * tgo))

            # dqz：侧向修正视线角速度。
            dqz = float(dz / (range_rate * tgo**2) + dzdt / (range_rate * tgo))

    # interceptor_theta：拦截弹当前航迹倾角。
    interceptor_theta = float(interceptor_state[4])

    # equilibrium_ny：当前航迹倾角下的纵向平衡项。
    equilibrium_ny = float(np.cos(interceptor_theta))

    # raw_ny_command：纵向比例导引指令。
    raw_ny_command = equilibrium_ny - float(config.navigation_constant) * range_rate * dqy / GRAVITY

    # raw_nz_command：侧向比例导引指令，并加入红方机动前馈补偿。
    raw_nz_command = (
        -float(config.navigation_constant) * range_rate * dqz / GRAVITY
        - float(config.compensation_gain) * float(red_lateral_overload)
    )

    # ny_command/nz_command：二维合成限幅后的指令。
    ny_command, nz_command = limit_planar_overload(
        ny_command=raw_ny_command,
        nz_command=raw_nz_command,
        max_overload=config.max_overload,
        theta=interceptor_theta,
    )

    # actual_pair_array：一阶自动驾驶仪响应后的实际双通道输出。
    actual_pair_array = FirstOrderAutopilot.compute_response(
        command=np.array([ny_command, nz_command], dtype=np.float64),
        previous_output=np.array(
            [previous_interceptor_ny_actual, previous_interceptor_nz_actual],
            dtype=np.float64,
        ),
        dt=dt,
        tau=config.tau,
    )

    # ny_actual/nz_actual：拆分自动驾驶仪输出。
    ny_actual = float(actual_pair_array[0])
    nz_actual = float(actual_pair_array[1])

    # ny_actual/nz_actual：实际输出也做一次合成限幅，保证响应模型不会越界。
    ny_actual, nz_actual = limit_planar_overload(
        ny_command=ny_actual,
        nz_command=nz_actual,
        max_overload=config.max_overload,
        theta=interceptor_theta,
    )

    # info：source_pn 诊断字段。
    info = {
        **relative_info,
        "source_range_rate": float(range_rate),
        "source_tgo": float(tgo),
        "source_dqy": float(dqy),
        "source_dqz": float(dqz),
        "raw_ny_command": float(raw_ny_command),
        "raw_nz_command": float(raw_nz_command),
        "ny_command": float(ny_command),
        "nz_command": float(nz_command),
        "ny_actual": float(ny_actual),
        "nz_actual": float(nz_actual),
    }

    return (float(ny_command), float(nz_command)), (float(ny_actual), float(nz_actual)), info


def differential_game_guidance_placeholder(*args, **kwargs):
    """
    预留微分对策制导律入口。

    参数：
        *args：
            预留位置参数。
        **kwargs：
            预留关键字参数。

    返回：
        不返回。当前函数始终抛出 NotImplementedError。
    """
    # error_message：明确说明该接口仍处于预留状态。
    error_message = "微分对策制导律接口将在后续阶段实现。"

    raise NotImplementedError(error_message)
