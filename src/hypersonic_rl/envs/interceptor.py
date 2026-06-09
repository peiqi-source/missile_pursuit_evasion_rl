"""
interceptor.py

作用：
    构建完整蓝方拦截弹模型。

当前实现：
    1. 单枚拦截弹；
    2. 中制导：比例导引项 + 弹道成型角误差反馈；
    3. 末制导：基于 tgo / 零控脱靶趋势的修正比例导引；
    4. 自动驾驶仪：ny / nz 两个通道均采用一阶动态；
    5. 过载限幅：对机动过载进行合成限幅；
    6. 状态更新：复用 update_point_mass_state()，保持坐标轴和符号一致。

注意：
    当前是“完整单枚拦截弹闭环”的第一版。
    后续可以在此基础上扩展：
        1. 两枚拦截弹；
        2. 更严格的弹道成型函数；
        3. 中末制导交接角约束；
        4. 真实拦截弹能量 / 高度 / 速度剖面。
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
    update_point_mass_state,
)
from hypersonic_rl.envs.guidance import apply_first_order_lag


def wrap_angle(angle: float) -> float:
    """
    将角度归一化到 [-pi, pi]。

    参数：
        angle：
            输入角度，单位 rad。

    返回：
        wrapped_angle：
            归一化后的角度。
    """
    wrapped_angle = (float(angle) + np.pi) % (2.0 * np.pi) - np.pi
    return float(wrapped_angle)


@dataclass
class InterceptorConfig:
    """
    InterceptorConfig

    作用：
        管理完整拦截弹参数。

    属性：
        max_overload：
            拦截弹最大可用机动过载。
            论文中拦截弹最大过载可取 8g，当前默认 8。

        tau：
            自动驾驶仪一阶响应时间常数。

        kill_radius：
            杀伤半径。

        midcourse_navigation_gain：
            中制导比例导引系数 N1。

        midcourse_theta_shaping_gain：
            中制导纵向弹道成型反馈增益。

        midcourse_psi_shaping_gain：
            中制导侧向弹道成型反馈增益。

        terminal_navigation_gain：
            末制导比例导引系数。
            一般应大于或等于中制导系数，用于提高末端命中精度。

        terminal_distance_threshold：
            距离小于该阈值时进入末制导。

        terminal_tgo_threshold：
            剩余时间小于该阈值时进入末制导。

        minimum_tgo：
            防止 tgo 过小导致数值爆炸。

        enable_vertical_channel：
            是否启用纵向 ny 通道。
            第一版默认启用，这样拦截弹是完整三维闭环。

        enable_lateral_channel：
            是否启用侧向 nz 通道。
    """

    max_overload: float = 8.0
    tau: float = 0.5
    kill_radius: float = 7.0

    midcourse_navigation_gain: float = 4.0
    midcourse_theta_shaping_gain: float = 1.5
    midcourse_psi_shaping_gain: float = 1.5

    terminal_navigation_gain: float = 6.0
    terminal_distance_threshold: float = 7000.0
    terminal_tgo_threshold: float = 4.0
    minimum_tgo: float = 0.2

    enable_vertical_channel: bool = True
    enable_lateral_channel: bool = True


@dataclass
class InterceptorControl:
    """
    InterceptorControl

    作用：
        保存拦截弹某一步的制导指令和实际控制量。

    属性：
        ny_command：
            纵向过载指令。

        nz_command：
            侧向过载指令。

        ny_actual：
            经过自动驾驶仪一阶环节后的纵向实际过载。

        nz_actual：
            经过自动驾驶仪一阶环节后的侧向实际过载。

        phase：
            当前制导阶段，取值为 "midcourse" 或 "terminal"。
    """

    ny_command: float
    nz_command: float
    ny_actual: float
    nz_actual: float
    phase: str


class Interceptor:
    """
    完整拦截弹对象。

    外部调用方式：
        interceptor.reset(initial_state)
        next_state, control, info = interceptor.step(target_state, dt)

    内部流程：
        target_state
            ↓
        计算相对运动信息
            ↓
        判断中制导 / 末制导
            ↓
        生成 ny / nz 过载指令
            ↓
        自动驾驶仪一阶响应
            ↓
        合成过载限幅
            ↓
        拦截弹状态更新
    """

    def __init__(self, config: InterceptorConfig) -> None:
        """
        初始化拦截弹。

        参数：
            config：
                拦截弹配置对象。
        """
        # config：保存拦截弹参数。
        self.config = config

        # state：拦截弹状态向量 [x, y, z, v, theta, psi, nx, ny, nz]。
        self.state = np.zeros(9, dtype=np.float64)

        # ny_actual：纵向实际过载。
        # theta=0 平飞时，ny=1 近似保持高度。
        self.ny_actual = 1.0

        # nz_actual：侧向实际过载。
        self.nz_actual = 0.0

        # ny_command：纵向过载指令。
        self.ny_command = 1.0

        # nz_command：侧向过载指令。
        self.nz_command = 0.0

        # phase：当前制导阶段。
        self.phase = "midcourse"

    def reset(self, initial_state: np.ndarray) -> None:
        """
        重置拦截弹状态。

        参数：
            initial_state：
                拦截弹初始状态向量，形状为 [9]。
        """
        # state：拷贝初始状态，避免外部数组被原地修改。
        self.state = np.asarray(initial_state, dtype=np.float64).copy()

        if self.state.shape[0] != 9:
            raise ValueError(f"拦截弹状态维度应为 9，实际为：{self.state.shape[0]}")

        # 从状态中恢复实际控制量。
        self.ny_actual = float(self.state[7])
        self.nz_actual = float(self.state[8])

        # 初始指令等于当前实际量。
        self.ny_command = self.ny_actual
        self.nz_command = self.nz_actual

        # 初始阶段为中制导。
        self.phase = "midcourse"

    def compute_interceptor_relative_info(self, target_state: np.ndarray) -> Dict[str, float]:
        """
        计算拦截弹相对目标的三维相对运动信息。

        参数：
            target_state：
                红方目标状态。

        返回：
            info：
                包含距离、接近速度、视线角、视线角速度、期望航迹角等信息。
        """
        # base_info：沿用项目中与源程序一致的相对位置定义。
        base_info = compute_relative_geometry(
            red_state=target_state,
            blue_state=self.state,
        )

        # dx/dy/dz：从拦截弹指向目标的相对位置。
        dx = float(base_info["dx"])
        dy = float(base_info["dy"])
        dz = float(base_info["dz"])

        # dxdt/dydt/dzdt：目标相对拦截弹速度。
        dxdt = float(base_info["dxdt"])
        dydt = float(base_info["dydt"])
        dzdt = float(base_info["dzdt"])

        # horizontal_distance：x-z 平面水平距离。
        horizontal_distance = float(np.sqrt(dx * dx + dz * dz) + EPS)

        # horizontal_distance_rate：水平距离变化率。
        horizontal_distance_rate = float((dx * dxdt + dz * dzdt) / horizontal_distance)

        # distance：三维距离。
        distance = float(base_info["distance"])

        # closing_speed：接近速度，>0 表示正在接近。
        closing_speed = float(base_info["closing_speed"])


        if closing_speed > EPS:
            # tgo：剩余飞行时间估计。
            tgo = distance / closing_speed
        else:
            # 如果没有接近，则给一个较大 tgo，避免直接进入末制导。
            tgo = 1e6

        # desired_theta：从拦截弹指向目标的期望视线倾角。
        desired_theta = float(np.arctan2(dy, horizontal_distance))

        # desired_psi：源程序坐标符号下，从拦截弹指向目标的期望航向角。
        # 因为 z_dot = -V cos(theta) sin(psi)，所以这里使用 atan2(-dz, dx)。
        desired_psi = float(np.arctan2(-dz, dx))

        # current_theta/current_psi：拦截弹当前航迹倾角和航向角。
        current_theta = float(self.state[4])
        current_psi = float(self.state[5])

        # theta_error：纵向视线角 / 弹道倾角误差。
        theta_error = wrap_angle(desired_theta - current_theta)

        # psi_error：侧向视线角 / 弹道偏角误差。
        psi_error = wrap_angle(desired_psi - current_psi)

        # lambda_theta_dot：纵向视线角速度。
        # lambda_theta = atan2(dy, rho)
        lambda_theta_dot = float(
            (horizontal_distance * dydt - dy * horizontal_distance_rate)
            / (horizontal_distance * horizontal_distance + dy * dy + EPS)
        )

        # lambda_psi_dot：侧向视线角速度。
        # lambda_psi = atan2(-dz, dx)
        lambda_psi_dot = float(
            (-dx * dzdt + dz * dxdt)
            / (dx * dx + dz * dz + EPS)
        )

        info = {
            **base_info,
            "horizontal_distance": horizontal_distance,
            "horizontal_distance_rate": horizontal_distance_rate,
            "tgo": float(tgo),
            "desired_theta": desired_theta,
            "desired_psi": desired_psi,
            "theta_error": theta_error,
            "psi_error": psi_error,
            "lambda_theta_dot": lambda_theta_dot,
            "lambda_psi_dot": lambda_psi_dot,
        }

        return info

    def select_phase(self, relative_info: Dict[str, float]) -> str:
        """
        判断当前制导阶段。

        参数：
            relative_info：
                相对运动信息。

        返回：
            phase：
                "midcourse" 或 "terminal"。
        """
        # distance：当前相对距离。
        distance = float(relative_info["distance"])

        # tgo：预计剩余飞行时间。
        tgo = float(relative_info["tgo"])

        if distance <= self.config.terminal_distance_threshold:
            return "terminal"

        if tgo <= self.config.terminal_tgo_threshold:
            return "terminal"

        return "midcourse"

    def compute_midcourse_guidance(self, relative_info: Dict[str, float]) -> Tuple[float, float]:
        """
        计算中制导过载指令。

        中制导结构：
            比例导引项 + 弹道成型项。

        工程表达：
            ny_cmd = cos(theta) + PN_y + shaping_y
            nz_cmd = PN_z + shaping_z

        返回：
            ny_command：
                纵向过载指令。

            nz_command：
                侧向过载指令。
        """
        # interceptor_speed：拦截弹速度。
        interceptor_speed = max(float(self.state[3]), EPS)

        # theta：当前航迹倾角。
        theta = float(self.state[4])

        # cos_theta：用于平飞平衡项。
        cos_theta = float(np.cos(theta))

        # closing_speed：接近速度。
        closing_speed = max(float(relative_info["closing_speed"]), 0.0)

        # tgo：剩余时间。
        tgo = max(float(relative_info["tgo"]), self.config.minimum_tgo)

        # lambda_theta_dot：纵向视线角速度。
        lambda_theta_dot = float(relative_info["lambda_theta_dot"])

        # lambda_psi_dot：侧向视线角速度。
        lambda_psi_dot = float(relative_info["lambda_psi_dot"])

        # theta_error：纵向弹道成型角误差。
        theta_error = float(relative_info["theta_error"])

        # psi_error：侧向弹道成型角误差。
        psi_error = float(relative_info["psi_error"])

        # pn_y：纵向比例导引项，单位为过载。
        pn_y = (
            float(self.config.midcourse_navigation_gain)
            * closing_speed
            * lambda_theta_dot
            / GRAVITY
        )

        # shaping_y：纵向弹道成型项。
        # 使用角误差 / tgo 构造期望角速度，再转化为过载。
        shaping_y = (
            interceptor_speed
            / GRAVITY
            * float(self.config.midcourse_theta_shaping_gain)
            * theta_error
            / tgo
        )

        # ny_command：纵向过载指令。
        # cos(theta) 是保持当前航迹倾角的平衡项。
        ny_command = cos_theta + pn_y + shaping_y

        # pn_z：侧向比例导引项。
        # 注意源程序坐标下 psi_dot = -g*nz/(V cos(theta))，
        # 所以侧向过载指令带负号。
        pn_z = (
            -float(self.config.midcourse_navigation_gain)
            * closing_speed
            * lambda_psi_dot
            / GRAVITY
        )

        # shaping_z：侧向弹道成型项。
        # 先用 psi_error / tgo 构造期望航向角速度，再由 psi_dot 反解 nz。
        shaping_z = (
            -interceptor_speed
            * max(np.cos(theta), 0.1)
            / GRAVITY
            * float(self.config.midcourse_psi_shaping_gain)
            * psi_error
            / tgo
        )

        # nz_command：侧向过载指令。
        nz_command = pn_z + shaping_z

        if not self.config.enable_vertical_channel:
            ny_command = cos_theta

        if not self.config.enable_lateral_channel:
            nz_command = 0.0

        return float(ny_command), float(nz_command)

    def compute_terminal_guidance(self, relative_info: Dict[str, float]) -> Tuple[float, float]:
        """
        计算末制导过载指令。

        当前实现：
            修正比例导引 Modified Proportional Navigation, MPN。

        设计思想：
            普通比例导引主要根据视线角速度生成制导指令；
            修正比例导引进一步引入距离变化率和剩余飞行时间 tgo，
            相当于根据当前相对位置和相对速度预测零控脱靶趋势，
            再生成更强的末端修正指令。

        与普通 / 高增益 PN 相比：
            1. 不只是简单增大导航系数；
            2. 显式使用 tgo；
            3. 对末端脱靶量更敏感；
            4. 更适合提高末制导命中精度。

        使用的源程序一致形式：
            v_r = range_rate

            tgo = -distance / v_r

            dqy = -dy / (v_r * tgo^2) - dydt / (v_r * tgo)
            dqz =  dz / (v_r * tgo^2) + dzdt / (v_r * tgo)

            ny_command = cos(theta) - N * v_r * dqy / g
            nz_command =             - N * v_r * dqz / g

        返回：
            ny_command：
                纵向过载指令。

            nz_command：
                侧向过载指令。
        """
        # theta：拦截弹当前航迹倾角。
        theta = float(self.state[4])

        # cos_theta：保持当前航迹倾角的平衡项。
        cos_theta = float(np.cos(theta))

        # distance：当前红蓝距离。
        distance = max(float(relative_info["distance"]), EPS)

        # v_r：距离变化率。
        # v_r < 0 表示拦截弹与目标正在接近。
        v_r = float(relative_info["closing_speed"])

        # dx/dy/dz：目标相对拦截弹的位置。
        dx = float(relative_info["dx"])
        dy = float(relative_info["dy"])
        dz = float(relative_info["dz"])

        # dxdt/dydt/dzdt：目标相对拦截弹的速度。
        dxdt = float(relative_info["dxdt"])
        dydt = float(relative_info["dydt"])
        dzdt = float(relative_info["dzdt"])

        # terminal_N：末制导修正比例导引系数。
        terminal_N = float(self.config.terminal_navigation_gain)

        if v_r >= -EPS:
            # 如果当前没有接近趋势，则退化为角误差修正。
            # 这种情况通常说明几何关系异常或已经错过交会窗口。
            tgo = self.config.minimum_tgo

            theta_error = float(relative_info["theta_error"])
            psi_error = float(relative_info["psi_error"])

            interceptor_speed = max(float(self.state[3]), EPS)
            cos_theta_safe = max(float(np.cos(theta)), 0.1)

            # ny_command：纵向角误差修正。
            ny_command = (
                    cos_theta
                    + interceptor_speed / GRAVITY
                    * theta_error
                    / max(tgo, self.config.minimum_tgo)
            )

            # nz_command：侧向角误差修正。
            # 根据 psi_dot = -g*nz/(V*cos(theta)) 反解 nz。
            nz_command = (
                    -interceptor_speed
                    * cos_theta_safe
                    / GRAVITY
                    * psi_error
                    / max(tgo, self.config.minimum_tgo)
            )

        else:
            # tgo：预计剩余交会时间。
            tgo = max(
                -distance / v_r,
                float(self.config.minimum_tgo),
            )

            # dqy：纵向修正视线角速度。
            # 该项综合了当前高度差 dy 和高度相对速度 dydt。
            dqy = float(
                -dy / (v_r * tgo ** 2)
                - dydt / (v_r * tgo)
            )

            # dqz：侧向修正视线角速度。
            # 该项综合了当前侧向差 dz 和侧向相对速度 dzdt。
            dqz = float(
                dz / (v_r * tgo ** 2)
                + dzdt / (v_r * tgo)
            )

            # ny_command：纵向修正比例导引。
            # cos_theta 是平衡项，后半部分用于修正末端纵向脱靶。
            ny_command = cos_theta - terminal_N * v_r * dqy / GRAVITY

            # nz_command：侧向修正比例导引。
            # 符号与源程序 source_pn 中 nzc_i = -N * v_r * dqz / g 保持一致。
            nz_command = -terminal_N * v_r * dqz / GRAVITY

        if not self.config.enable_vertical_channel:
            ny_command = cos_theta

        if not self.config.enable_lateral_channel:
            nz_command = 0.0

        return float(ny_command), float(nz_command)

    def limit_overload(self, ny_command: float, nz_command: float) -> Tuple[float, float]:
        """
        对过载指令进行合成限幅。

        参数：
            ny_command：
                纵向过载指令。

            nz_command：
                侧向过载指令。

        返回：
            limited_ny：
                限幅后的纵向过载。

            limited_nz：
                限幅后的侧向过载。

        说明：
            为了更物理，限幅对象不是 ny 本身，
            而是相对于平衡项 cos(theta) 的机动过载分量。
        """
        # theta：当前航迹倾角。
        theta = float(self.state[4])

        # equilibrium_ny：保持当前航迹倾角所需的近似平衡项。
        equilibrium_ny = float(np.cos(theta))

        # delta_ny：纵向机动过载。
        delta_ny = float(ny_command) - equilibrium_ny

        # delta_nz：侧向机动过载。
        delta_nz = float(nz_command)

        # maneuver_norm：合成机动过载。
        maneuver_norm = float(np.sqrt(delta_ny * delta_ny + delta_nz * delta_nz))

        # max_overload：最大允许机动过载。
        max_overload = float(self.config.max_overload)

        if maneuver_norm <= max_overload:
            return float(ny_command), float(nz_command)

        # scale：缩放比例。
        scale = max_overload / max(maneuver_norm, EPS)

        # limited_delta_ny / limited_delta_nz：限幅后的机动过载。
        limited_delta_ny = delta_ny * scale
        limited_delta_nz = delta_nz * scale

        # limited_ny / limited_nz：恢复到总过载指令。
        limited_ny = equilibrium_ny + limited_delta_ny
        limited_nz = limited_delta_nz

        return float(limited_ny), float(limited_nz)

    def apply_autopilot(self, ny_command: float, nz_command: float, dt: float) -> Tuple[float, float]:
        """
        自动驾驶仪一阶动态响应。

        参数：
            ny_command：
                纵向过载指令。

            nz_command：
                侧向过载指令。

            dt：
                仿真步长。

        返回：
            ny_actual：
                纵向实际过载。

            nz_actual：
                侧向实际过载。
        """
        # ny_actual：纵向一阶滞后响应。
        ny_actual = apply_first_order_lag(
            command=ny_command,
            previous_output=self.ny_actual,
            dt=dt,
            tau=self.config.tau,
        )

        # nz_actual：侧向一阶滞后响应。
        nz_actual = apply_first_order_lag(
            command=nz_command,
            previous_output=self.nz_actual,
            dt=dt,
            tau=self.config.tau,
        )

        # 再做一次合成过载限幅，保证实际输出也满足约束。
        ny_actual, nz_actual = self.limit_overload(ny_actual, nz_actual)

        self.ny_actual = float(ny_actual)
        self.nz_actual = float(nz_actual)

        return self.ny_actual, self.nz_actual

    def step(self, target_state: np.ndarray, dt: float) -> Tuple[np.ndarray, InterceptorControl, Dict[str, float]]:
        """
        推进一步完整拦截弹闭环。

        参数：
            target_state：
                红方目标状态。

            dt：
                仿真步长。

        返回：
            next_state：
                更新后的拦截弹状态。

            control：
                当前步控制量。

            info：
                诊断信息字典。
        """
        # relative_info：当前相对运动信息。
        relative_info = self.compute_interceptor_relative_info(target_state)

        # phase：中末制导阶段选择。
        self.phase = self.select_phase(relative_info)

        if self.phase == "midcourse":
            # 中制导：比例导引 + 弹道成型项。
            ny_command, nz_command = self.compute_midcourse_guidance(relative_info)
        elif self.phase == "terminal":
            # 末制导：增强比例导引。
            ny_command, nz_command = self.compute_terminal_guidance(relative_info)
        else:
            raise ValueError(f"未知拦截弹制导阶段：{self.phase}")

        # 指令限幅。
        ny_command, nz_command = self.limit_overload(ny_command, nz_command)

        # 保存指令。
        self.ny_command = float(ny_command)
        self.nz_command = float(nz_command)

        # 自动驾驶仪一阶响应。
        ny_actual, nz_actual = self.apply_autopilot(
            ny_command=ny_command,
            nz_command=nz_command,
            dt=dt,
        )

        # 拦截弹状态更新。
        self.state = update_point_mass_state(
            state=self.state,
            nx=0.0,
            ny=ny_actual,
            nz=nz_actual,
            dt=dt,
        )

        # 同步状态中的控制量字段。
        self.state[6] = 0.0
        self.state[7] = ny_actual
        self.state[8] = nz_actual

        # control：当前控制量对象。
        control = InterceptorControl(
            ny_command=float(ny_command),
            nz_command=float(nz_command),
            ny_actual=float(ny_actual),
            nz_actual=float(nz_actual),
            phase=self.phase,
        )

        # info：诊断信息。
        info = {
            **relative_info,
            "interceptor_phase": self.phase,
            "blue_ny_command": float(ny_command),
            "blue_nz_command": float(nz_command),
            "blue_ny_actual": float(ny_actual),
            "blue_nz_actual": float(nz_actual),
            "blue_command_overload": float(nz_command),
            "blue_inertial_overload": float(nz_actual),
            "blue_speed": float(self.state[3]),
            "blue_theta": float(self.state[4]),
            "blue_psi": float(self.state[5]),
        }

        return self.state.copy(), control, info