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
    当前代码先实现简化模型，保证 SAC / LSTM-SAC 训练闭环稳定可运行。

    当前支持三种积分方法：
        1. semi_implicit_euler：
            半隐式欧拉 / 辛欧拉法。
            先更新速度和角度，再用更新后的速度和角度推进位置。

        2. explicit_euler：
            显式欧拉法。
            使用当前时刻速度和角度推进位置。

        3. rk4：
            四阶龙格库达法。
            对连续时间三自由度质点方程做四阶积分，更接近论文仿真设置。

    后续如果需要更贴近论文，可以继续在本文件中接入：
        1. 高超声速飞行器气动模型；
        2. 拦截弹气动模型；
        3. 论文完整三自由度动力学模型。
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from hypersonic_rl.envs.integrators import rk4_step


# GRAVITY：标准重力加速度，单位 m/s^2。
GRAVITY = 9.81

# SUPPORTED_INTEGRATION_MODES：当前质点模型支持的积分模式。
SUPPORTED_INTEGRATION_MODES = {"semi_implicit_euler", "explicit_euler", "rk4"}

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


def simple_point_mass_derivative(
    state_vector: np.ndarray,
    control_vector: np.ndarray,
) -> np.ndarray:
    """
    当前简化三自由度质点模型的连续时间状态导数。

    该函数是为了给 RK4 积分器使用。

    状态向量：
        state_vector = [x, y, z, v, theta, psi]

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

        psi：
            航向角，单位 rad。

    控制向量：
        control_vector = [nx, ny, nz]

    其中：
        nx：
            速度方向过载倍数。

        ny：
            法向过载倍数。

        nz：
            侧向过载倍数。

    连续动力学方程：
        x_dot     = v * cos(theta) * cos(psi)
        y_dot     = v * sin(theta)
        z_dot     = -v * cos(theta) * sin(psi)

        v_dot     = g * (nx - sin(theta))
        theta_dot = g * (ny - cos(theta)) / v
        psi_dot   = -g * nz / (v * cos(theta))

    注意：
        这里的 nx、ny、nz 都是“过载倍数”，不是 m/s^2。
        因此在动力学中需要乘以 GRAVITY。

    返回：
        state_dot：
            状态导数 [x_dot, y_dot, z_dot, v_dot, theta_dot, psi_dot]。
    """
    # state_vector：确保使用 float64，降低长时间仿真的数值误差。
    state_vector = np.asarray(state_vector, dtype=np.float64)

    # control_vector：控制量向量。
    control_vector = np.asarray(control_vector, dtype=np.float64)

    # x, y, z：位置状态。
    # 当前导数计算中 x/y/z 不直接参与方程，但保留解包是为了保持状态定义清晰。
    x = float(state_vector[0])
    y = float(state_vector[1])
    z = float(state_vector[2])

    # velocity_scalar：速度大小。
    velocity_scalar = max(float(state_vector[3]), EPS)

    # theta：航迹倾角。
    theta = float(state_vector[4])

    # psi：航向角。
    psi = float(state_vector[5])

    # nx, ny, nz：三个方向过载倍数。
    nx = float(control_vector[0])
    ny = float(control_vector[1])
    nz = float(control_vector[2])

    # 防止 psi_dot 分母过小。
    cos_theta_safe = float(np.clip(np.cos(theta), 0.1, 1.0))

    # x_dot：x 方向位置导数。
    x_dot = velocity_scalar * np.cos(theta) * np.cos(psi)

    # y_dot：高度方向位置导数。
    y_dot = velocity_scalar * np.sin(theta)

    # z_dot：侧向位置导数。
    z_dot = -velocity_scalar * np.cos(theta) * np.sin(psi)

    # v_dot：速度导数。
    v_dot = GRAVITY * (nx - np.sin(theta))

    # theta_dot：航迹倾角导数。
    theta_dot = GRAVITY * (ny - np.cos(theta)) / velocity_scalar

    # psi_dot：航向角导数。
    # 注意这里是负号，与 z_dot = -V cos(theta) sin(psi) 的坐标系约定保持一致。
    psi_dot = -GRAVITY * nz / (velocity_scalar * cos_theta_safe)

    # state_dot：连续时间状态导数。
    state_dot = np.array(
        [
            x_dot,
            y_dot,
            z_dot,
            v_dot,
            theta_dot,
            psi_dot,
        ],
        dtype=np.float64,
    )

    return state_dot


def update_point_mass_state(
    state: np.ndarray,
    nx: float,
    ny: float,
    nz: float,
    dt: float,
    integration_mode: str = "semi_implicit_euler",
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
            SAC / LSTM-SAC 输出的机动动作主要作用在该方向。

        dt：
            仿真积分步长，单位 s。

        integration_mode：
            数值积分方法。

            可选：
                "semi_implicit_euler"：
                    半隐式欧拉 / 辛欧拉法。
                    先更新速度和角度，再用更新后的速度和角度推进位置。

                "explicit_euler"：
                    显式欧拉法。
                    使用当前速度和角度推进位置。

                "rk4"：
                    四阶龙格库达法。
                    对连续时间三自由度质点方程进行四阶积分。

    返回：
        next_state：
            更新后的状态向量，形状为 [9]。

    说明：
        当前简化模型为：
            v_dot     = g * (nx - sin(theta))
            theta_dot = g * (ny - cos(theta)) / v
            psi_dot   = -g * nz / (v cos(theta))

        这样做的好处是：
            1. 当 theta=0 且 ny=1 时，高度近似保持；
            2. nz 可以直接改变航向角 psi；
            3. 适合当前端到端 SAC / LSTM-SAC 先建立训练闭环。

        与旧版本相比：
            本函数新增了 rk4 分支，用于和论文中的四阶龙格库达仿真设置对齐。
    """
    # current_state：复制当前状态，避免原地修改输入。
    current_state = np.asarray(state, dtype=np.float64).copy()

    # integration_mode：检查积分模式是否合法。
    if integration_mode not in SUPPORTED_INTEGRATION_MODES:
        raise ValueError(
            f"integration_mode 必须属于 {sorted(SUPPORTED_INTEGRATION_MODES)}，当前为：{integration_mode}"
        )

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

    # ------------------------------------------------------------
    # RK4 分支
    # ------------------------------------------------------------
    # RK4 不是简单地“先更新速度再更新位置”，而是把完整的连续方程：
    #     state_dot = f(state, control)
    # 在一个 dt 内计算四次斜率，再加权平均得到下一状态。
    #
    # 当前状态只对前 6 维 [x, y, z, v, theta, psi] 做动力学积分；
    # aux_1、aux_2、aux_3 仍然作为预留状态位原样保留。
    # ------------------------------------------------------------
    if integration_mode == "rk4":
        # state_vector：取出参与动力学积分的前 6 维状态。
        state_vector = np.array(
            [
                x,
                y,
                z,
                velocity_scalar,
                theta,
                psi,
            ],
            dtype=np.float64,
        )

        # control_vector：当前步内假设控制量保持不变。
        # 这是一种常见离散控制系统处理方式：
        #     智能体每个 step 输出一次动作；
        #     在该 step 内控制量保持常值；
        #     动力学用 RK4 对状态进行高精度推进。
        control_vector = np.array(
            [
                float(nx),
                float(ny),
                float(nz),
            ],
            dtype=np.float64,
        )

        # next_state_vector：RK4 积分后的前 6 维状态。
        next_state_vector = rk4_step(
            derivative_fn=simple_point_mass_derivative,
            state=state_vector,
            control=control_vector,
            dt=dt,
        )

        # next_state：保留原状态长度和 aux 预留位。
        next_state = current_state.copy()
        next_state[0] = float(next_state_vector[0])
        next_state[1] = float(next_state_vector[1])
        next_state[2] = float(next_state_vector[2])
        next_state[3] = max(float(next_state_vector[3]), EPS)
        next_state[4] = float(next_state_vector[4])
        next_state[5] = float(next_state_vector[5])

        return next_state.astype(np.float64)

    # ------------------------------------------------------------
    # 欧拉积分分支
    # ------------------------------------------------------------
    # 下面保留原工程已有逻辑：
    #     1. 先用当前状态计算 v_dot / theta_dot / psi_dot；
    #     2. 再根据积分模式选择位置更新使用旧速度还是新速度。
    # ------------------------------------------------------------

    # v_dot：速度变化率。
    v_dot = GRAVITY * (float(nx) - np.sin(theta))

    # theta_dot：航迹倾角变化率。
    theta_dot = GRAVITY * (float(ny) - np.cos(theta)) / velocity_scalar

    # cos_theta_safe：防止 cos(theta) 接近 0 导致除零。

    cos_theta_safe = float(np.clip(np.cos(theta), 0.1, 1.0))

    # psi_dot：航向角变化率。
    # 注意这里是负号，与坐标系 z_dot = -V cos(theta) sin(psi) 保持一致。
    psi_dot = -GRAVITY * float(nz) / (velocity_scalar * cos_theta_safe)

    # next_velocity_scalar：下一时刻速度。
    next_velocity_scalar = max(velocity_scalar + v_dot * dt, EPS)

    # next_theta：下一时刻航迹倾角。
    next_theta = theta + theta_dot * dt

    # next_psi：下一时刻航向角。
    next_psi = psi + psi_dot * dt

    # temp_state：用于计算位置更新的临时状态。
    # semi_implicit_euler：
    #     保持旧工程行为，先更新速度/角度，再用新速度/角度推进位置。
    #
    # explicit_euler：
    #     使用当前速度/角度推进位置，便于和教材中的显式欧拉法对照。
    temp_state = current_state.copy()
    if integration_mode == "semi_implicit_euler":
        temp_state[3] = next_velocity_scalar
        temp_state[4] = next_theta
        temp_state[5] = next_psi

    # velocity_vector：用于推进位置的速度矢量。
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


def compute_relative_geometry(red_state: np.ndarray, interceptor_state: np.ndarray) -> Dict[str, float]:
    """
    计算红蓝双方在 x-z 平面内的相对几何关系。

    参数：
        red_state：
            红方状态向量。

        interceptor_state：
            拦截弹状态向量。

    返回：
        distance：
            三维相对距离，单位 m。

        closing_speed：
            接近速度，单位 m/s。
            closing_speed > 0 表示双方正在接近。

        range_rate：
            距离变化率，单位 m/s。
            range_rate < 0 表示双方正在接近。

        los_angle：
            x-z 平面视线角，单位 rad。

        los_rate_standard：
            标准二维视线角速度，单位 rad/s。
    """
    # red_position：红方三维位置。
    red_position = np.asarray(red_state[:3], dtype=np.float64)

    # interceptor_position：拦截弹三维位置。
    interceptor_position = np.asarray(interceptor_state[:3], dtype=np.float64)

    # relative_position：从拦截弹指向红方的相对位置向量。
    relative_position = red_position - interceptor_position

    # red_velocity：红方速度矢量。
    red_velocity = build_velocity_vector(red_state)

    # interceptor_velocity：拦截弹速度矢量。
    interceptor_velocity = build_velocity_vector(interceptor_state)

    # relative_velocity：红方相对拦截弹的速度。
    relative_velocity = red_velocity - interceptor_velocity

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

    # los_rate_standard：标准二维视线角速度。
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