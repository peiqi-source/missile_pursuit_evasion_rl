"""
integrators.py

作用：
    管理连续时间动力学方程的数值积分方法。

工程背景：
    论文仿真中提到采用四阶龙格库达法（RK4）推进动力学方程。
    当前工程原本只支持显式欧拉和半隐式欧拉/辛欧拉，
    因此本文件用于新增一个通用 RK4 积分器。

设计原则：
    1. 积分器只负责“如何从当前状态推进到下一状态”；
    2. 动力学方程本身由外部传入的 derivative_fn 提供；
    3. 这样后续无论是当前简化三自由度模型，还是论文气动三自由度模型，
       都可以复用同一个 RK4 积分器。
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def rk4_step(
    derivative_fn: Callable[[np.ndarray, Any], np.ndarray],
    state: np.ndarray,
    control: Any,
    dt: float,
) -> np.ndarray:
    """
    使用四阶龙格库达法推进一个仿真步长。

    参数：
        derivative_fn：
            连续时间动力学导数函数。

            函数形式为：
                derivative_fn(state, control) -> state_dot

            其中：
                state：
                    当前状态向量。

                control：
                    当前控制量。

                state_dot：
                    当前状态导数。

        state：
            当前状态向量。

            对当前简化三自由度质点模型而言，状态为：
                [x, y, z, v, theta, psi]

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

        control：
            当前控制量。

            对当前简化三自由度质点模型而言，控制量为：
                [nx, ny, nz]

            其中：
                nx：
                    速度方向过载倍数。

                ny：
                    法向过载倍数。

                nz：
                    侧向过载倍数。

            注意：
                这里统一使用“过载倍数”，不是 m/s^2。
                例如 nz = 2.0 表示 2g。

        dt：
            仿真积分步长，单位 s。

    返回：
        next_state：
            RK4 积分后的下一时刻状态向量。

    数学形式：
        k1 = f(x_k, u_k)
        k2 = f(x_k + 0.5 dt k1, u_k)
        k3 = f(x_k + 0.5 dt k2, u_k)
        k4 = f(x_k + dt k3, u_k)

        x_{k+1} = x_k + dt / 6 * (k1 + 2k2 + 2k3 + k4)

    工程说明：
        与欧拉法相比，RK4 在高速交会场景下对位置、视线角、
        最近距离和脱靶量的计算更稳定，也更贴近论文仿真设置。
    """
    current_state = np.asarray(state, dtype=np.float64)

    k1 = derivative_fn(current_state, control)
    k2 = derivative_fn(current_state + 0.5 * dt * k1, control)
    k3 = derivative_fn(current_state + 0.5 * dt * k2, control)
    k4 = derivative_fn(current_state + dt * k3, control)

    next_state = current_state + (dt / 6.0) * (
        k1 + 2.0 * k2 + 2.0 * k3 + k4
    )

    return next_state.astype(np.float64)