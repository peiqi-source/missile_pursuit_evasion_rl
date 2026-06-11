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
from typing import Any, Dict, Optional, Tuple

import numpy as np

from hypersonic_rl.envs.autopilot import FirstOrderAutopilot, limit_planar_overload
from hypersonic_rl.envs.dynamics import (
    EPS,
    compute_relative_geometry,
    update_point_mass_state,
)
from hypersonic_rl.envs.guidance import (
    GuidanceContext,
    ProportionalNavigationConfig,
    compute_guidance_command,
)


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

        target_compensation_gain：
            对目标侧向机动的前馈补偿系数。
            该项与 source_pn 中的红方机动补偿保持同一符号约定。

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
    kill_radius: float = 5.0

    midcourse_navigation_gain: float = 4.0
    midcourse_theta_shaping_gain: float = 1.5
    midcourse_psi_shaping_gain: float = 0.0
    target_compensation_gain: float = 4.0

    terminal_navigation_gain: float = 6.0
    terminal_distance_threshold: float = 10000.0
    terminal_tgo_threshold: float = 4.0
    minimum_tgo: float = 0.2

    enable_vertical_channel: bool = True
    enable_lateral_channel: bool = True
    autopilot_rate_limit: Optional[float] = None
    dynamics_integration_mode: str = "semi_implicit_euler"


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
        next_state, control, info = interceptor.step(target_state, dt, target_lateral_overload)

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

    def __init__(
            self,
            config: InterceptorConfig,
            guidance_mode: str = "mid_terminal_interceptor",
            navigation_config: Optional[ProportionalNavigationConfig] = None,
    ) -> None:
        """
        初始化拦截弹。

        参数：
            config：
                拦截弹配置对象。
        """
        # config：保存拦截弹参数。
        self.config = config

        # guidance_mode：当前拦截弹使用的制导模式。
        # 说明：
        #     这里保留现有 guidance_mode 字段名，不新增复杂命名。
        #     后续新增制导方法时，优先在 guidance.py 的统一入口中扩展。
        self.guidance_mode = str(guidance_mode)

        # navigation_config：source_pn 使用的配置。
        # 说明：
        #     mid_terminal_interceptor 不依赖该配置；
        #     source_pn 需要该配置计算比例导引、一阶自动驾驶仪响应和限幅。
        self.navigation_config = navigation_config

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

        # autopilot：蓝方拦截弹双通道一阶自动驾驶仪。
        self.autopilot = FirstOrderAutopilot(
            tau=self.config.tau,
            initial_output=np.array([self.ny_actual, self.nz_actual], dtype=np.float64),
            rate_limit=self.config.autopilot_rate_limit,
        )

        # last_guidance_components：保存最近一步制导律各分项，便于定位 PN、成型项或末制导 ZEM 是否主导。
        self.last_guidance_components: Dict[str, Any] = {}

        # last_autopilot_info：保存最近一步自动驾驶仪限速/限幅诊断。
        self.last_autopilot_info: Dict[str, Any] = {}

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
        # 如果外部初始状态没有正确设置 ny，则默认使用 cos(theta) 作为平飞平衡项。
        initial_theta = float(self.state[4])
        initial_ny = float(self.state[7])
        initial_nz = float(self.state[8])

        if abs(initial_ny) < 1e-8:
            initial_ny = float(np.cos(initial_theta))

        self.ny_actual = initial_ny
        self.nz_actual = initial_nz

        # autopilot：重置双通道自动驾驶仪内部输出，使其与拦截弹状态一致。
        self.autopilot.reset(
            initial_output=np.array([self.ny_actual, self.nz_actual], dtype=np.float64)
        )

        self.state[7] = self.ny_actual
        self.state[8] = self.nz_actual

        # 初始指令等于当前实际量。
        self.ny_command = self.ny_actual
        self.nz_command = self.nz_actual

        # 初始阶段为中制导。
        self.phase = "midcourse"

        # 诊断缓存：每次 reset 后清空上一回合的制导和自动驾驶仪诊断。
        self.last_guidance_components = {}
        self.last_autopilot_info = {}

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
            interceptor_state=self.state,
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


    def step(
            self,
            target_state: np.ndarray,
            dt: float,
            target_lateral_overload: float = 0.0,
    ) -> Tuple[np.ndarray, InterceptorControl, Dict[str, float]]:
        """
        推进单枚拦截弹一步。

        参数：
            track：
                单枚拦截弹跟踪记录。

            red_state：
                红方当前状态。

            red_lateral_overload：
                红方一阶自动驾驶仪之后的实际横向过载。

            dt：
                仿真步长。

        返回：
            guidance_info：
                单枚拦截弹返回的制导诊断信息。

        说明：
            该函数不区分具体制导模式。
            具体制导模式由 track.interceptor.step() 内部根据 guidance_mode 分发。
        """

        # relative_info：当前相对运动信息。
        relative_info = self.compute_interceptor_relative_info(target_state)

        # guidance_config：当前制导模式使用的配置对象。
        if self.guidance_mode == "source_pn":
            if self.navigation_config is None:
                raise ValueError("source_pn 模式需要提供 navigation_config。")
            guidance_config = self.navigation_config
            active_max_overload = float(self.navigation_config.max_overload)
            dynamics_integration_mode = self.navigation_config.dynamics_integration_mode
            target_compensation_gain = float(self.navigation_config.compensation_gain)
        else:
            guidance_config = self.config
            active_max_overload = float(self.config.max_overload)
            dynamics_integration_mode = self.config.dynamics_integration_mode
            target_compensation_gain = float(self.config.target_compensation_gain)

        # previous_phase：记录上一时刻阶段，用于诊断阶段切换。
        previous_phase = self.phase

        # context：统一制导上下文。
        # 这里把相对几何信息传给 guidance.py，由统一入口完成阶段判断和指令计算。
        context = GuidanceContext(
            red_state=target_state,
            interceptor_state=self.state,
            red_lateral_overload=float(target_lateral_overload),
            previous_ny_actual=float(self.ny_actual),
            previous_nz_actual=float(self.nz_actual),
            dt=float(dt),
            current_time=0.0,
            relative_info=relative_info,
        )

        # guidance_command：统一制导入口。
        # step() 不再直接判断中制导/末制导，也不再调用额外包装函数。
        guidance_command = compute_guidance_command(
            guidance_mode=self.guidance_mode,
            context=context,
            config=guidance_config,
        )

        # phase：当前实际制导阶段。
        self.phase = str(guidance_command.phase)

        # phase_changed：标记本步是否发生阶段切换。
        phase_changed = bool(previous_phase != self.phase)

        # ny_command/nz_command：统一入口输出的原始过载指令。
        ny_command = float(guidance_command.ny_command)
        nz_command = float(guidance_command.nz_command)

        # last_guidance_components：保存制导分项，后续打包进 info。
        self.last_guidance_components = dict(guidance_command.info)

        # raw_nz_command：记录加入目标机动补偿前的侧向制导律输出。
        raw_ny_command = float(ny_command)
        raw_nz_command = float(nz_command)

        # target_compensation：目标侧向机动前馈补偿。
        target_compensation = -target_compensation_gain * float(target_lateral_overload)

        # nz_command：加入目标机动前馈补偿后的侧向指令。
        nz_command = float(nz_command) + float(target_compensation)

        # compensated_nz_command：记录目标机动前馈后的侧向原始指令。
        compensated_nz_command = float(nz_command)

        # active_max_overload：当前制导模式对应的过载上限。
        if self.guidance_mode == "source_pn":
            if self.navigation_config is None:
                raise ValueError("source_pn 模式需要提供 navigation_config。")
            active_max_overload = float(self.navigation_config.max_overload)
        else:
            active_max_overload = float(self.config.max_overload)

        # ny_command/nz_command：直接调用 autopilot.py 中的二维合成过载限幅函数。
        ny_command, nz_command = limit_planar_overload(
            ny_command=float(ny_command),
            nz_command=float(nz_command),
            max_overload=active_max_overload,
            theta=float(self.state[4]),
        )

        # command_delta：记录完整拦截弹指令限幅裁剪量。
        command_delta = np.array(
            [raw_ny_command - ny_command, compensated_nz_command - nz_command],
            dtype=np.float64,
        )

        # command_saturated：任一通道被二维合成限幅裁剪时置 True。
        command_saturated = bool(np.any(np.abs(command_delta) > 1e-10))

        # equilibrium_ny：当前航迹倾角下维持平衡所需的纵向过载。
        equilibrium_ny = float(np.cos(float(self.state[4])))

        # planar_raw_norm：目标补偿后的原始指令机动幅值。
        planar_raw_norm = float(
            np.sqrt((raw_ny_command - equilibrium_ny) ** 2 + compensated_nz_command ** 2)
        )

        planar_saturation_ratio = planar_raw_norm / max(active_max_overload, EPS)

        # 保存指令。
        self.ny_command = float(ny_command)
        self.nz_command = float(nz_command)

        # autopilot_actual_pair：一阶自动驾驶仪响应后的实际过载。
        autopilot_actual_pair, autopilot_info = self.autopilot.step_with_info(
            command=np.array([ny_command, nz_command], dtype=np.float64),
            dt=dt,
        )

        # last_autopilot_info：保存自动驾驶仪诊断信息。
        self.last_autopilot_info = dict(autopilot_info)

        # ny_actual/nz_actual：拆分自动驾驶仪输出。
        ny_actual = float(autopilot_actual_pair[0])
        nz_actual = float(autopilot_actual_pair[1])

        # pre_limit_actual_pair：记录实际过载限幅前的输出。
        pre_limit_actual_pair = np.array([ny_actual, nz_actual], dtype=np.float64)

        # ny_actual/nz_actual：对实际输出再做一次二维合成过载限幅。
        # 说明：
        #     这里直接调用 autopilot.py 中的 limit_planar_overload()，
        #     不再通过 Interceptor.limit_overload() 或 apply_autopilot() 包装。
        ny_actual, nz_actual = limit_planar_overload(
            ny_command=float(ny_actual),
            nz_command=float(nz_actual),
            max_overload=float(active_max_overload),
            theta=float(self.state[4]),
        )

        # actual_limit_delta：记录自动驾驶仪输出经过最终二维限幅时的裁剪量。
        actual_limit_delta = pre_limit_actual_pair - np.array(
            [ny_actual, nz_actual],
            dtype=np.float64,
        )

        # actual_output_saturated：实际输出是否触发二维合成限幅。
        actual_output_saturated = bool(np.any(np.abs(actual_limit_delta) > 1e-10))

        # output_saturated：合并自动驾驶仪内部硬限幅和二维合成限幅诊断。
        self.last_autopilot_info["output_saturated"] = bool(
            self.last_autopilot_info.get("output_saturated", False)
            or actual_output_saturated
        )

        # 保存实际过载。
        self.ny_actual = float(ny_actual)
        self.nz_actual = float(nz_actual)

        # 将自动驾驶仪内部状态同步到最终限幅后的实际输出。
        self.autopilot.reset(
            initial_output=np.array([self.ny_actual, self.nz_actual], dtype=np.float64)
        )
        self.autopilot.last_info = dict(self.last_autopilot_info)

        # 拦截弹状态更新。
        self.state = update_point_mass_state(
            state=self.state,
            nx=0.0,
            ny=ny_actual,
            nz=nz_actual,
            dt=dt,
            integration_mode=dynamics_integration_mode,
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
        info: Dict[str, float] ={
            **relative_info,
            "interceptor_phase": self.phase,
            "interceptor_previous_phase": previous_phase,
            "interceptor_phase_changed": phase_changed,
            "ny_command": float(ny_command),
            "nz_command": float(nz_command),
            "ny_actual": float(ny_actual),
            "nz_actual": float(nz_actual),
            "raw_ny_command": float(raw_ny_command),
            "raw_nz_command": float(raw_nz_command),
            "compensated_raw_nz_command": float(compensated_nz_command),
            "planar_raw_norm": float(planar_raw_norm),
            "planar_saturation_ratio": float(planar_saturation_ratio),
            "command_saturated": command_saturated,
            "autopilot_rate_saturated": bool(self.last_autopilot_info.get("rate_saturated", False)),
            "autopilot_output_saturated": bool(self.last_autopilot_info.get("output_saturated", False)),
            "interceptor_target_lateral_overload": float(target_lateral_overload),
            "interceptor_target_compensation": float(target_compensation),
            "interceptor_target_compensation_gain": float(target_compensation_gain),
            **self.last_guidance_components,
            "speed": float(self.state[3]),
            "theta": float(self.state[4]),
            "psi": float(self.state[5]),
        }

        return self.state.copy(), control, info
