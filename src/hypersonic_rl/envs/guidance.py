"""制导律模块，包含当前比例导引实现和论文扩展接口。"""

from __future__ import annotations

from typing import Any

import numpy as np

from hypersonic_rl.envs.dynamics import apply_first_order_lag, compute_relative_velocity


def proportional_navigation_guidance(
    red_state: np.ndarray,
    blue_state: np.ndarray,
    red_lateral_acceleration: np.ndarray | float,
    previous_blue_output: np.ndarray | float,
    dt: float,
    tau_i: float,
    nzc_i_max: float,
    navigation_constant: float,
    gravity: float = 9.81,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """计算蓝方比例导引控制量。

    参数:
        red_state: 红方飞行器状态向量。
        blue_state: 蓝方拦截弹状态向量。
        red_lateral_acceleration: 红方经过一阶惯性后的横向过载。
        previous_blue_output: 蓝方上一时刻一阶惯性输出。
        dt: 仿真积分步长。
        tau_i: 蓝方拦截弹制导控制一阶惯性时间常数。
        nzc_i_max: 蓝方最大横向过载约束。
        navigation_constant: 比例导引导航系数。
        gravity: 重力加速度。

    返回:
        control_vector: 蓝方动力学使用的三轴过载向量 [0, 1, nz]。
        blue_command: 蓝方比例导引原始横向过载指令。
        inertial_blue_output: 蓝方经过一阶惯性环节后的横向过载。
        guidance_info: 视线角速度等中间量。
    """
    # relative_position：红方相对蓝方的位置向量。
    relative_position = np.asarray(red_state[:3], dtype=np.float64) - np.asarray(
        blue_state[:3], dtype=np.float64
    )
    # relative_velocity：红方相对蓝方的速度向量。
    relative_velocity = compute_relative_velocity(red_state, blue_state)
    # distance：红蓝双方相对距离。
    distance = max(float(np.linalg.norm(relative_position)), 1e-9)
    # closing_velocity：沿视线方向的相对速度，原始代码记为 v_r。
    closing_velocity_raw = float(np.dot(relative_position, relative_velocity) / distance)
    closing_velocity_sign = 1.0 if closing_velocity_raw >= 0.0 else -1.0
    closing_velocity = closing_velocity_sign * max(abs(closing_velocity_raw), 1e-9)
    # time_to_go：按当前闭合速度估计的剩余飞行时间。
    time_to_go_raw = -distance / closing_velocity
    time_to_go_sign = 1.0 if time_to_go_raw >= 0.0 else -1.0
    time_to_go = time_to_go_sign * max(abs(time_to_go_raw), 1e-9)

    dx, dy, dz = relative_position
    dxdt, dydt, dzdt = relative_velocity
    _ = dx, dxdt
    # dqy：高低视线角速度，当前奖励暂未直接使用。
    dqy = -dy / (closing_velocity * time_to_go**2) - dydt / (
        closing_velocity * time_to_go
    )
    # dqz：方位视线角速度，用于蓝方比例导引和奖励分析。
    dqz = dz / (closing_velocity * time_to_go**2) + dzdt / (
        closing_velocity * time_to_go
    )

    red_lateral = float(np.asarray(red_lateral_acceleration).reshape(-1)[0])
    blue_command = -navigation_constant * closing_velocity * dqz / gravity - 0.5 * red_lateral
    blue_command = np.asarray(
        np.clip(blue_command, -nzc_i_max, nzc_i_max), dtype=np.float64
    ).reshape(1)
    inertial_blue_output = apply_first_order_lag(
        previous_blue_output, blue_command, dt=dt, tau=tau_i
    ).reshape(1)
    control_vector = np.array([0.0, 1.0, float(inertial_blue_output[0])], dtype=np.float64)
    guidance_info = {
        "distance": distance,
        "closing_velocity": closing_velocity,
        "time_to_go": time_to_go,
        "dqy": float(dqy),
        "dqz": float(dqz),
        "blue_command": float(blue_command[0]),
        "inertial_blue_command": float(inertial_blue_output[0]),
    }
    return control_vector, blue_command, inertial_blue_output, guidance_info


def differential_game_guidance(*args: Any, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """预留微分对策制导律接口。

    后续将依据论文第 3、4 章补充微分对策制导律，当前阶段不在普通 SAC
    工程化范围内实现。
    """
    raise NotImplementedError("微分对策制导律将在后续阶段依据论文第 3、4 章实现。")


def differential_game_parameterized_guidance(
    *args: Any, **kwargs: Any
) -> tuple[np.ndarray, dict[str, Any]]:
    """预留 SAC 输出制导律参数的接口。

    论文第 4 章中 SAC 输出的是微分对策制导律导航系数/增益参数，再由制导律生成
    最终过载指令。当前端到端 SAC 仍直接输出红方横向过载，因此这里只保留清晰接口。
    """
    raise NotImplementedError("参数化微分对策制导律将在第 4 章映射扩展中实现。")
