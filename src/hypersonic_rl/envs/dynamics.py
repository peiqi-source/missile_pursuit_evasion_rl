"""
dynamics.py

作用：
    管理高超声速飞行器和拦截弹的简化三自由度质点运动学更新。

状态向量定义：
    state = [x, y, z, v, theta, psi, aux_1, aux_2, aux_3]

其中：
    x：
        前向距离坐标，单位 m。

    y：
        高度坐标，单位 m。

    z：
        侧向距离坐标，单位 m。

    v：
        飞行速度，单位 m/s。

    theta：
        航迹倾角，单位 rad。
        theta = 0 表示水平飞行。

    psi：
        航向角，单位 rad。
        psi = 0 表示沿 x 正方向飞行。
        psi = pi 表示沿 x 负方向飞行。

    aux_1, aux_2, aux_3：
        预留状态位，当前阶段不参与运动学更新。
        后续可以扩展为攻角、侧滑角、滚转角等变量。

工程说明：
    当前代码先实现简化模型，保证 SAC 训练闭环稳定可运行。
    后续如果需要更贴近论文，可以在本文件中替换为更高保真的飞行动力学模型。
"""

from __future__ import annotations

from typing import Dict

import numpy as np


# GRAVITY：重力加速度，单位 m/s^2。
GRAVITY = 9.8

# EPS：防止除零的小常数。
EPS = 1e-8


def degrees_to_radians(angle_deg: float) -> float:
    """
    将角度制转换为弧度制。

    参数：
        angle_deg：
            角度值，单位 degree。

    返回：
        angle_rad：
            弧度值，单位 rad。
    """
    # angle_rad：转换后的弧度。
    angle_rad = float(angle_deg) * np.pi / 180.0
    return angle_rad


def build_velocity_vector(state: np.ndarray) -> np.ndarray:
    """
    根据状态向量计算速度矢量。

    参数：
        state：
            飞行器状态向量，至少包含 [x, y, z, v, theta, psi]。

    返回：
        velocity：
            三维速度向量 [vx, vy, vz]，单位 m/s。
    """
    # velocity_scalar：速度大小 v。
    velocity_scalar = float(state[3])

    # theta：航迹倾角。
    theta = float(state[4])

    # psi：航向角。
    psi = float(state[5])

    # vx：x 方向速度分量。
    vx = velocity_scalar * np.cos(theta) * np.cos(psi)

    # vy：y 方向速度分量，即高度方向速度。
    vy = velocity_scalar * np.sin(theta)

    # vz：z 方向速度分量，即侧向速度。
    # 与原始复现代码保持一致：z_dot = -V * cos(theta) * sin(psi)。
    vz = -velocity_scalar * np.cos(theta) * np.sin(psi)

    # velocity：速度矢量。
    velocity = np.array([vx, vy, vz], dtype=np.float64)

    return velocity


def update_point_mass_state(
    state: np.ndarray,
    nx: float,
    ny: float,
    nz: float,
    dt: float,
) -> np.ndarray:
    """
    使用简化三自由度质点模型更新飞行器状态。

    参数：
        state：
            当前状态向量，形状为 [9]。

        nx：
            速度方向过载控制量。
            当前阶段通常设为 0，表示不主动改变速度。

        ny：
            法向过载控制量。
            当前阶段通常设为 1，用于近似维持高度。

        nz：
            侧向过载控制量。
            SAC 输出的机动动作主要作用在该方向。

        dt：
            仿真积分步长，单位 s。

    返回：
        next_state：
            更新后的状态向量，形状为 [9]。

    说明：
        这里采用工程简化模型：
            v_dot     = g * (nx - sin(theta))
            theta_dot = g / v * (ny - cos(theta))
            psi_dot   = g / (v cos(theta)) * nz

        这样做的好处是：
            1. 当 theta=0 且 ny=1 时，高度近似保持；
            2. nz 可以直接改变航向角 psi；
            3. 适合当前端到端 SAC 先建立训练闭环。
    """
    # current_state：复制当前状态，避免原地修改输入。
    current_state = np.asarray(state, dtype=np.float64).copy()

    if current_state.shape[0] < 6:
        raise ValueError(f"state 至少需要 6 个元素，当前长度为：{current_state.shape[0]}")

    # x, y, z：当前位置。
    x = float(current_state[0])
    y = float(current_state[1])
    z = float(current_state[2])

    # velocity_scalar：速度大小。
    velocity_scalar = max(float(current_state[3]), EPS)

    # theta：航迹倾角。
    theta = float(current_state[4])

    # psi：航向角。
    psi = float(current_state[5])

    # v_dot：速度变化率。
    v_dot = GRAVITY * (float(nx) - np.sin(theta))

    # theta_dot：航迹倾角变化率。
    theta_dot = GRAVITY * (float(ny) - np.cos(theta)) / velocity_scalar

    # cos_theta_safe：防止 cos(theta) 接近 0 导致除零。
    cos_theta_safe = np.clip(np.cos(theta), 0.1, 1.0)

    # psi_dot：航向角变化率。
    psi_dot = -GRAVITY * float(nz) / (velocity_scalar * cos_theta_safe)

    # next_velocity_scalar：下一时刻速度。
    next_velocity_scalar = max(velocity_scalar + v_dot * dt, EPS)

    # next_theta：下一时刻航迹倾角。
    next_theta = theta + theta_dot * dt

    # next_psi：下一时刻航向角。
    next_psi = psi + psi_dot * dt

    # 使用更新后的速度和角度计算位置变化。
    # temp_state：临时状态，用于计算下一时刻速度矢量。
    temp_state = current_state.copy()
    temp_state[3] = next_velocity_scalar
    temp_state[4] = next_theta
    temp_state[5] = next_psi

    # velocity_vector：更新后的速度矢量。
    velocity_vector = build_velocity_vector(temp_state)

    # next_x, next_y, next_z：下一时刻位置。
    next_x = x + velocity_vector[0] * dt
    next_y = y + velocity_vector[1] * dt
    next_z = z + velocity_vector[2] * dt

    # next_state：输出状态。
    next_state = current_state.copy()
    next_state[0] = next_x
    next_state[1] = next_y
    next_state[2] = next_z
    next_state[3] = next_velocity_scalar
    next_state[4] = next_theta
    next_state[5] = next_psi

    return next_state.astype(np.float64)


def compute_relative_geometry(red_state: np.ndarray, blue_state: np.ndarray) -> Dict[str, float]:
    """
    计算红蓝双方在 x-z 平面内的相对几何关系。

    参数：
        red_state：
            红方状态向量。

        blue_state：
            蓝方状态向量。

    返回：
        distance：
            三维相对距离，单位 m。

        closing_speed：
            接近速度，单位 m/s。
            closing_speed > 0 表示双方正在接近。

        los_angle：
            x-z 平面视线角，单位 rad。

        los_rate：
            x-z 平面视线角速度，单位 rad/s。
    """
    # red_position：红方三维位置。
    red_position = np.asarray(red_state[:3], dtype=np.float64)

    # blue_position：蓝方三维位置。
    blue_position = np.asarray(blue_state[:3], dtype=np.float64)

    # relative_position：从蓝方指向红方的相对位置向量。
    relative_position = red_position - blue_position

    # red_velocity：红方速度矢量。
    red_velocity = build_velocity_vector(red_state)

    # blue_velocity：蓝方速度矢量。
    blue_velocity = build_velocity_vector(blue_state)

    # relative_velocity：红方相对蓝方的速度。
    relative_velocity = red_velocity - blue_velocity

    # distance：三维距离。
    distance = float(np.linalg.norm(relative_position))

    # range_rate：距离变化率，range_rate < 0 表示双方正在接近。
    range_rate = float(np.dot(relative_position, relative_velocity)) / max(distance, EPS)

    # closing_speed：接近速度，closing_speed > 0 表示双方正在接近。
    closing_speed = -range_rate

    # dx/dz：x-z 平面相对位置。
    dx = float(relative_position[0])
    dz = float(relative_position[2])

    # dxdt/dzdt：x-z 平面相对速度。
    dxdt = float(relative_velocity[0])
    dzdt = float(relative_velocity[2])

    # los_angle：x-z 平面视线角。
    los_angle = float(np.arctan2(dz, dx))

    # los_rate_standard：标准二维视线角速度，仅用于诊断，不直接替代源程序 dqz。
    los_rate_standard = float((dx * dzdt - dz * dxdt) / (dx * dx + dz * dz + EPS))

    info = {
        "distance": distance,
        "closing_speed": closing_speed,
        "range_rate": range_rate,
        "dx": dx,
        "dy": float(relative_position[1]),
        "dz": dz,
        "dxdt": dxdt,
        "dydt": float(relative_velocity[1]),
        "dzdt": dzdt,
        "los_angle": los_angle,
        "los_rate_standard": los_rate_standard,
    }

    return info
