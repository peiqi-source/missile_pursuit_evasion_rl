"""
autopilot.py

作用：
    统一管理飞行器和拦截弹自动驾驶仪相关的动态响应模型。

当前实现：
    1. 一阶自动驾驶仪 FirstOrderAutopilot；
    2. 二维机动过载合成限幅函数；
    3. 兼容标量控制量和向量控制量的一阶响应计算。

工程说明：
    论文结构中制导律和自动驾驶仪是两个独立环节。制导律负责给出期望过载，
    自动驾驶仪负责把期望过载变成有动态滞后的实际过载。把这部分集中到
    本文件后，后续扩展二阶自动驾驶仪、带速率限制的响应模型或真实弹体
    约束时，不需要在环境、比例导引和完整拦截弹模型中重复修改。
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from hypersonic_rl.envs.dynamics import EPS


class FirstOrderAutopilot:
    """
    一阶自动驾驶仪。

    作用：
        根据一阶惯性环节模拟控制指令到实际过载输出的动态滞后。

    参数：
        tau：
            一阶惯性时间常数，单位 s。数值越大，响应越慢。
        initial_output：
            自动驾驶仪初始输出，可以是标量，也可以是向量。

    返回：
        该类本身不直接返回结果；调用 step() 时返回当前时刻的一阶响应输出。

    公式：
        output_new = output_old + (command - output_old) * dt / tau
    """

    def __init__(self, tau: float, initial_output: Any = 0.0) -> None:
        """
        初始化一阶自动驾驶仪。

        参数：
            tau：
                一阶惯性时间常数，单位 s。
            initial_output：
                初始实际输出，支持 float 或 numpy 数组。

        返回：
            None。
        """
        # tau：保存自动驾驶仪响应时间常数，并避免后续除零。
        self.tau = max(float(tau), EPS)

        # output：保存当前自动驾驶仪实际输出。
        self.output = np.asarray(initial_output, dtype=np.float64).copy()

    @staticmethod
    def compute_response(command: Any, previous_output: Any, dt: float, tau: float) -> np.ndarray:
        """
        计算一阶惯性环节单步响应。

        参数：
            command：
                当前制导律或智能体给出的期望控制量。
            previous_output：
                上一仿真步的一阶惯性实际输出。
            dt：
                仿真步长，单位 s。
            tau：
                一阶惯性时间常数，单位 s。

        返回：
            current_output：
                当前仿真步的一阶惯性实际输出，numpy 数组形式。
        """
        # command_array：把输入指令统一转换为 numpy 数组，便于标量和向量共用。
        command_array = np.asarray(command, dtype=np.float64)

        # previous_array：把上一时刻输出统一转换为 numpy 数组。
        previous_array = np.asarray(previous_output, dtype=np.float64)

        # safe_tau：防止 tau 过小导致除零或响应发散。
        safe_tau = max(float(tau), EPS)

        # current_output：按照一阶惯性公式计算当前输出。
        current_output = previous_array + (command_array - previous_array) * float(dt) / safe_tau

        return np.asarray(current_output, dtype=np.float64)

    def reset(self, initial_output: Any = 0.0) -> None:
        """
        重置自动驾驶仪内部输出。

        参数：
            initial_output：
                重置后的实际输出，支持 float 或 numpy 数组。

        返回：
            None。
        """
        # output：覆盖内部状态，使下一步响应从指定输出开始。
        self.output = np.asarray(initial_output, dtype=np.float64).copy()

    def step(self, command: Any, dt: float) -> np.ndarray:
        """
        推进一步自动驾驶仪响应。

        参数：
            command：
                当前仿真步期望控制量。
            dt：
                仿真步长，单位 s。

        返回：
            output：
                当前仿真步的一阶惯性实际输出。
        """
        # output：用当前指令和上一输出计算新的一阶响应，并写回内部状态。
        self.output = self.compute_response(
            command=command,
            previous_output=self.output,
            dt=dt,
            tau=self.tau,
        )

        return self.output.copy()


def limit_planar_overload(
    ny_command: float,
    nz_command: float,
    max_overload: float,
    theta: float,
) -> Tuple[float, float]:
    """
    对纵向和侧向机动过载做二维合成限幅。

    参数：
        ny_command：
            纵向总过载指令。
        nz_command：
            侧向过载指令。
        max_overload：
            允许的最大机动过载幅值。
        theta：
            当前航迹倾角，单位 rad。

    返回：
        limited_ny：
            限幅后的纵向总过载。
        limited_nz：
            限幅后的侧向过载。

    说明：
        纵向通道包含维持当前航迹倾角的平衡项 cos(theta)。真正消耗机动能力的是
        ny - cos(theta)，因此合成限幅对象是 [ny-cos(theta), nz]。
    """
    # equilibrium_ny：当前航迹倾角下维持平衡所需的近似纵向过载。
    equilibrium_ny = float(np.cos(float(theta)))

    # delta_ny：纵向机动过载分量，不包含平衡项。
    delta_ny = float(ny_command) - equilibrium_ny

    # delta_nz：侧向机动过载分量。
    delta_nz = float(nz_command)

    # maneuver_norm：二维合成机动过载幅值。
    maneuver_norm = float(np.sqrt(delta_ny * delta_ny + delta_nz * delta_nz))

    # max_overload_value：最大允许机动过载。
    max_overload_value = float(max_overload)

    if maneuver_norm <= max_overload_value:
        return float(ny_command), float(nz_command)

    # scale：超过约束时按比例缩放两个机动分量，保持方向不变。
    scale = max_overload_value / max(maneuver_norm, EPS)

    # limited_delta_ny：限幅后的纵向机动分量。
    limited_delta_ny = delta_ny * scale

    # limited_delta_nz：限幅后的侧向机动分量。
    limited_delta_nz = delta_nz * scale

    # limited_ny：把纵向平衡项加回总过载。
    limited_ny = equilibrium_ny + limited_delta_ny

    # limited_nz：侧向通道本身就是机动分量。
    limited_nz = limited_delta_nz

    return float(limited_ny), float(limited_nz)
