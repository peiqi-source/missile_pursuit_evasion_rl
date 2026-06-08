"""追逃场景中的质点动力学和相对运动工具函数。"""

from __future__ import annotations

import numpy as np


def state_to_velocity(state: np.ndarray) -> np.ndarray:
    """将飞行器状态向量转换为惯性系速度向量。

    参数:
        state: 飞行器状态向量，格式为 [x, y, z, speed, theta, psi, nx, ny, nz]。

    返回:
        速度向量 [vx, vy, vz]，单位为 m/s。
    """
    # speed：飞行器速度大小。
    speed = float(state[3])
    # theta：弹道倾角，决定速度在垂直方向上的分量。
    theta = float(state[4])
    # psi：弹道偏角，决定水平面内的速度方向。
    psi = float(state[5])
    vx = speed * np.cos(theta) * np.cos(psi)
    vy = speed * np.sin(theta)
    vz = -speed * np.cos(theta) * np.sin(psi)
    return np.array([vx, vy, vz], dtype=np.float64)


def update_point_mass_state(
    state: np.ndarray,
    control: np.ndarray,
    dt: float,
    gravity: float = 9.81,
) -> np.ndarray:
    """根据三自由度质点模型更新飞行器状态。

    参数:
        state: 当前飞行器状态向量，格式为 [x, y, z, speed, theta, psi, nx, ny, nz]。
        control: 当前过载控制向量 [nx, ny, nz]。
        dt: 仿真积分步长，单位为 s。
        gravity: 重力加速度，单位为 m/s^2。

    返回:
        更新后的飞行器状态向量。
    """
    next_state = np.asarray(state, dtype=np.float64).copy()
    # nx：切向过载，影响速度大小变化。
    nx = float(control[0])
    # ny：法向过载，影响弹道倾角变化。
    ny = float(control[1])
    # nz：侧向过载，影响弹道偏角变化。
    nz = float(control[2])
    next_state[6:9] = np.array([nx, ny, nz], dtype=np.float64)

    # speed：飞行器速度大小，加入下限避免分母接近 0。
    speed = max(float(next_state[3]), 1e-6)
    # theta：弹道倾角。
    theta = float(next_state[4])
    # psi：弹道偏角。
    psi = float(next_state[5])

    speed = speed + gravity * (nx - np.sin(theta)) * dt
    speed = max(speed, 1e-6)
    theta = theta + gravity / speed * (ny - np.cos(theta)) * dt
    cos_theta = np.cos(theta)
    cos_theta = np.sign(cos_theta) * max(abs(cos_theta), 1e-6)
    psi = psi - gravity / (speed * cos_theta) * nz * dt

    next_state[3] = speed
    next_state[4] = theta
    next_state[5] = psi
    next_state[0] = next_state[0] + speed * np.cos(theta) * np.cos(psi) * dt
    next_state[1] = next_state[1] + speed * np.sin(theta) * dt
    next_state[2] = next_state[2] - speed * np.cos(theta) * np.sin(psi) * dt
    return next_state


def apply_first_order_lag(
    previous_output: np.ndarray | float,
    command: np.ndarray | float,
    dt: float,
    tau: float,
) -> np.ndarray:
    """计算一阶惯性环节输出。

    参数:
        previous_output: 上一时刻一阶惯性环节输出。
        command: 当前控制指令。
        dt: 仿真积分步长。
        tau: 一阶惯性时间常数，数值越大表示响应越慢。

    返回:
        当前时刻一阶惯性环节输出。
    """
    previous = np.asarray(previous_output, dtype=np.float64)
    current_command = np.asarray(command, dtype=np.float64)
    if tau <= 0:
        return current_command.copy()
    return previous + dt * (current_command - previous) / tau


def compute_relative_distance(red_state: np.ndarray, blue_state: np.ndarray) -> float:
    """计算红蓝双方三维相对距离。

    参数:
        red_state: 红方飞行器状态向量。
        blue_state: 蓝方拦截弹状态向量。

    返回:
        红蓝双方欧氏距离，单位为 m。
    """
    relative_position = np.asarray(red_state[:3]) - np.asarray(blue_state[:3])
    return float(np.linalg.norm(relative_position))


def compute_relative_velocity(red_state: np.ndarray, blue_state: np.ndarray) -> np.ndarray:
    """计算红方相对蓝方的速度向量。

    参数:
        red_state: 红方飞行器状态向量。
        blue_state: 蓝方拦截弹状态向量。

    返回:
        相对速度向量 v_red - v_blue，单位为 m/s。
    """
    return state_to_velocity(red_state) - state_to_velocity(blue_state)

