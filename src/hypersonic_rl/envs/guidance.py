"""
guidance.py

作用：
    管理 source_pn 等工程制导律计算。

当前约定：
    1. 红方仍由 SAC/LSTM-SAC 输出 1 维横向过载动作；
    2. 蓝方拦截弹制导律输出内部 ny / nz 双通道过载；
    3. source_pn 使用论文/legacy 风格的修正比例导引公式，并导出 tgo、dqy/dqz、PN 分项和饱和诊断；
    4. 第 4 章微分对策解析制导不是当前 active baseline，只保留结构化状态入口。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from hypersonic_rl.envs.dynamics import EPS, GRAVITY, compute_relative_geometry


# SUPPORTED_GUIDANCE_MODES：集中保存当前支持的蓝方制导模式。
# 后续新增制导方法时，优先在这里登记，避免多个文件里重复手写字符串。
SUPPORTED_GUIDANCE_MODES = {
    "source_pn",
    "mid_terminal_interceptor",
}


@dataclass
class GuidanceCommand:
    """
    GuidanceCommand

    作用：
        统一保存制导律输出结果。

    设计目的：
        不同制导方法的内部计算过程可以不同，
        但最终都只输出期望过载指令。
        自动驾驶仪响应、过载限幅和状态更新统一交给 Interceptor.step() 处理。

    属性：
        ny_command：
            纵向过载指令，单位为 g 倍数。

        nz_command：
            侧向过载指令，单位为 g 倍数。

        phase：
            当前制导阶段或制导模式。
            例如：
                source_pn
                midcourse
                terminal

        info：
            制导诊断信息。
            用于后续写入 info、CSV 或绘图分析。
    """

    ny_command: float
    nz_command: float
    phase: str
    info: Dict[str, Any]


@dataclass
class GuidanceContext:
    """
    GuidanceContext

    作用：
        统一保存制导律计算所需的上下文信息。

    设计目的：
        不同制导方法需要的输入略有差异。
        如果每个函数都单独写一长串参数，后续会很难维护。
        因此这里把常用输入集中放在一个上下文对象里。

    属性：
        red_state：
            红方当前状态向量。

        interceptor_state：
            当前拦截弹状态向量。

        red_lateral_overload：
            红方一阶自动驾驶仪之后的实际横向过载。

        previous_ny_actual：
            拦截弹上一仿真步纵向实际过载。

        previous_nz_actual：
            拦截弹上一仿真步侧向实际过载。

        dt：
            仿真步长，单位 s。

        current_time：
            当前仿真时间，单位 s。
            如果调用方暂时没有传入，可以使用默认值 0.0。

        relative_info：
            可选相对几何信息。
            如果调用方已经计算过相对几何，可以传入这里，避免重复计算。
    """

    red_state: np.ndarray
    interceptor_state: np.ndarray
    red_lateral_overload: float
    previous_ny_actual: float
    previous_nz_actual: float
    dt: float
    current_time: float = 0.0
    relative_info: Optional[Dict[str, Any]] = None




@dataclass
class ProportionalNavigationConfig:
    """
    source_pn 比例导引配置。

    属性：
        navigation_constant：
            比例导引系数 N。
        max_overload：
            拦截弹最大可用机动过载，单位 g。
        tau：
            拦截弹一阶自动驾驶仪时间常数，单位 s。
        compensation_gain：
            对红方横向机动的前馈补偿系数。
        autopilot_rate_limit：
            可选自动驾驶仪输出变化率限制，单位 g/s。
        dynamics_integration_mode：
            拦截弹质点运动学积分模式。
    """

    navigation_constant: float = 4.0
    max_overload: float = 6.0
    tau: float = 0.5
    compensation_gain: float = 0.5
    autopilot_rate_limit: Optional[float] = None
    dynamics_integration_mode: str = "semi_implicit_euler"


def compute_interceptor_proportional_navigation(
    red_state: np.ndarray,
    interceptor_state: np.ndarray,
    config: ProportionalNavigationConfig,
    relative_info: Optional[Dict[str, Any]] = None,
) -> GuidanceCommand:
    """
    计算 source_pn 模式下单枚拦截弹的双通道制导指令。

    参数：
        red_state：
            红方当前状态向量。

        interceptor_state：
            拦截弹当前状态向量。

        config：
            source_pn 比例导引配置。

        relative_info：
            可选相对几何信息。
            如果外部已经计算过相对几何，可以传入这里，避免重复计算。

    返回：
        guidance_command：
            统一格式的制导指令。

    说明：
        本函数只负责计算期望过载指令：
            ny_command
            nz_command

        不再负责：
            1. 目标机动前馈补偿；
            2. 二维合成过载限幅；
            3. 自动驾驶仪一阶响应；
            4. 状态更新。

        这些闭环执行步骤统一由 Interceptor.step() 完成。
    """
    # relative_info：如果调用方没有传入，则在这里计算。
    if relative_info is None:
        relative_info = compute_relative_geometry(
            red_state=red_state,
            interceptor_state=interceptor_state,
        )

    # dy/dz：纵向和侧向相对位置分量。
    dy = float(relative_info["dy"])
    dz = float(relative_info["dz"])

    # dydt/dzdt：纵向和侧向相对速度分量。
    dydt = float(relative_info["dydt"])
    dzdt = float(relative_info["dzdt"])

    # distance：三维相对距离。
    distance = max(float(relative_info["distance"]), EPS)

    # range_rate：距离变化率，小于 0 表示双方正在接近。
    range_rate = float(relative_info["range_rate"])

    if abs(range_rate) < EPS:
        # tgo/dqy/dqz：距离变化率过小时不计算剩余飞行时间和修正视线角速度。
        tgo = 0.0
        dqy = 0.0
        dqz = 0.0
    else:
        # tgo：预计交会时间。
        tgo = float(-distance / range_rate)

        if abs(tgo) < EPS:
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

    # source_pn_y：纵向比例导引分量。
    source_pn_y = -float(config.navigation_constant) * range_rate * dqy / GRAVITY

    # source_pn_z：侧向比例导引分量。
    source_pn_z = -float(config.navigation_constant) * range_rate * dqz / GRAVITY

    # ny_command/nz_command：这里只输出制导律本身的期望过载。
    # 目标机动补偿会在 Interceptor.apply_target_compensation() 中统一加入。
    ny_command = equilibrium_ny + source_pn_y
    nz_command = source_pn_z

    # info：source_pn 诊断字段。
    info: Dict[str, Any] = {
        **relative_info,
        "source_range_rate": float(range_rate),
        "source_tgo": float(tgo),
        "source_dqy": float(dqy),
        "source_dqz": float(dqz),
        "source_pn_y": float(source_pn_y),
        "source_pn_z": float(source_pn_z),
        "guidance_raw_ny_command": float(ny_command),
        "guidance_raw_nz_command": float(nz_command),
        "raw_ny_command": float(ny_command),
        "raw_nz_command": float(nz_command),
    }

    return GuidanceCommand(
        ny_command=float(ny_command),
        nz_command=float(nz_command),
        phase="source_pn",
        info=info,
    )



def differential_game_guidance_placeholder(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    返回第 4 章微分对策制导律的非激活状态说明。

    参数：
        *args：
            预留位置参数。
        **kwargs：
            预留关键字参数。

    返回：
        status：
            结构化状态字典，明确该入口不是当前第 5 章端到端训练的 active baseline。
    """
    # status：用结构化返回替代异常，避免工程脚本误把占位入口当成失败链路。
    status = {
        "active": False,
        "baseline_name": "chapter4_differential_game_guidance",
        "reason": "本轮训练与 benchmark 以第 5 章端到端 SAC/LSTM-SAC 为 active baseline，第 4 章解析制导仅保留为后续扩展项。",
    }

    return status


def compute_guidance_command(
    guidance_mode: str,
    context: GuidanceContext,
    config: Any,
) -> GuidanceCommand:
    """
    统一计算蓝方制导指令。

    参数：
        guidance_mode：
            制导模式。

            当前支持：
                source_pn：
                    source_pn：
                    使用单阶段比例导引，返回期望过载指令。

                mid_terminal_interceptor：
                    根据当前距离和剩余时间选择中制导或末制导，
                    并返回统一格式的指令过载。

        context：
            制导计算上下文。

        config：
            当前制导模式对应的配置对象。

    返回：
        guidance_command：
            统一格式的制导输出。
    """
    # mode：统一转成字符串，避免外部传入枚举或其他对象导致判断异常。
    mode = str(guidance_mode)

    if mode == "source_pn":
        return compute_interceptor_proportional_navigation(
            red_state=context.red_state,
            interceptor_state=context.interceptor_state,
            config=config,
            relative_info=context.relative_info,
        )

    if mode == "mid_terminal_interceptor":
        if context.relative_info is None:
            raise ValueError(
                "mid_terminal_interceptor 需要 context.relative_info。"
                "请先在 Interceptor 中计算相对运动信息后再传入。"
            )

        guidance_command = compute_mid_terminal_command(
            interceptor_state=context.interceptor_state,
            relative_info=context.relative_info,
            config=config,
            phase=None,
        )

        return guidance_command

    raise ValueError(
        f"未知制导模式：{guidance_mode}，当前支持：{sorted(SUPPORTED_GUIDANCE_MODES)}"
    )


def compute_midcourse_command(
    interceptor_state: np.ndarray,
    relative_info: Dict[str, float],
    config: Any,
) -> GuidanceCommand:
    """
    计算中制导过载指令。

    中制导结构：
        比例导引项 + 弹道成型项。

    指令形式：
        ny_command = cos(theta) + pn_y + shaping_y
        nz_command = pn_z + shaping_z

    参数：
        interceptor_state：
            当前拦截弹状态向量：
                [x, y, z, v, theta, psi, nx, ny, nz]

        relative_info：
            当前相对运动信息。

            需要包含：
                closing_speed：
                    接近速度。

                tgo：
                    预计剩余飞行时间。

                lambda_theta_dot：
                    纵向视线角速度。

                lambda_psi_dot：
                    侧向视线角速度。

                theta_error：
                    纵向角误差。

                psi_error：
                    侧向角误差。

        config：
            拦截弹配置对象。

            需要包含：
                midcourse_navigation_gain
                midcourse_theta_shaping_gain
                midcourse_psi_shaping_gain
                minimum_tgo
                enable_vertical_channel
                enable_lateral_channel

    返回：
        guidance_command：
            统一格式制导指令。
    """
    # interceptor_speed：拦截弹速度。
    interceptor_speed = max(float(interceptor_state[3]), EPS)

    # theta：当前航迹倾角。
    theta = float(interceptor_state[4])

    # cos_theta：用于平飞平衡项。
    cos_theta = float(np.cos(theta))

    # closing_speed：接近速度。
    closing_speed = max(float(relative_info["closing_speed"]), 0.0)

    # tgo：剩余时间，设置下限避免成型项数值爆炸。
    tgo = max(float(relative_info["tgo"]), float(config.minimum_tgo))

    # lambda_theta_dot：纵向视线角速度。
    lambda_theta_dot = float(relative_info["lambda_theta_dot"])

    # lambda_psi_dot：侧向视线角速度。
    lambda_psi_dot = float(relative_info["lambda_psi_dot"])

    # theta_error：纵向弹道成型角误差。
    theta_error = float(relative_info["theta_error"])

    # psi_error：侧向弹道成型角误差。
    psi_error = float(relative_info["psi_error"])

    # pn_y：纵向比例导引项，单位为过载倍数。
    pn_y = (
        float(config.midcourse_navigation_gain)
        * closing_speed
        * lambda_theta_dot
        / GRAVITY
    )

    # shaping_y：纵向弹道成型项。
    # 使用角误差 / tgo 构造期望角速度，再转化为过载。
    shaping_y = (
        interceptor_speed
        / GRAVITY
        * float(config.midcourse_theta_shaping_gain)
        * theta_error
        / tgo
    )

    # ny_command：纵向过载指令。
    # cos(theta) 是维持当前航迹倾角的平衡项。
    ny_command = cos_theta + pn_y + shaping_y

    # pn_z：侧向比例导引项。
    # 当前坐标系下：
    #     psi_dot = -g * nz / (V cos(theta))
    # 所以侧向过载指令带负号。
    pn_z = (
        -float(config.midcourse_navigation_gain)
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
        * float(config.midcourse_psi_shaping_gain)
        * psi_error
        / tgo
    )

    # nz_command：侧向过载指令。
    nz_command = pn_z + shaping_z

    if not bool(config.enable_vertical_channel):
        ny_command = cos_theta

    if not bool(config.enable_lateral_channel):
        nz_command = 0.0

    # info：保存中制导各分项，便于后续调参、画图和写 CSV。
    info: Dict[str, Any] = {
        "guidance_phase": "midcourse",
        "midcourse_pn_y": float(pn_y),
        "midcourse_shaping_y": float(shaping_y),
        "midcourse_pn_z": float(pn_z),
        "midcourse_shaping_z": float(shaping_z),
        "midcourse_tgo": float(tgo),
        "midcourse_lambda_theta_dot": float(lambda_theta_dot),
        "midcourse_lambda_psi_dot": float(lambda_psi_dot),
        "guidance_raw_ny_command": float(ny_command),
        "guidance_raw_nz_command": float(nz_command),
    }

    return GuidanceCommand(
        ny_command=float(ny_command),
        nz_command=float(nz_command),
        phase="midcourse",
        info=info,
    )


def compute_terminal_command(
    interceptor_state: np.ndarray,
    relative_info: Dict[str, float],
    config: Any,
) -> GuidanceCommand:
    """
    计算末制导过载指令。

    当前末制导形式：
        使用距离变化率、剩余飞行时间和相对位置/速度估计末端脱靶趋势，
        再生成纵向和侧向修正指令。

    使用形式：
        v_r = range_rate

        tgo = -distance / v_r

        dqy = -dy / (v_r * tgo^2) - dydt / (v_r * tgo)
        dqz =  dz / (v_r * tgo^2) + dzdt / (v_r * tgo)

        ny_command = cos(theta) - N * v_r * dqy / g
        nz_command =             - N * v_r * dqz / g

    参数：
        interceptor_state：
            当前拦截弹状态向量。

        relative_info：
            当前相对运动信息。

        config：
            拦截弹配置对象。

    返回：
        guidance_command：
            统一格式制导指令。
    """
    # theta：拦截弹当前航迹倾角。
    theta = float(interceptor_state[4])

    # cos_theta：保持当前航迹倾角的平衡项。
    cos_theta = float(np.cos(theta))

    # distance：当前红蓝距离。
    distance = max(float(relative_info["distance"]), EPS)

    # v_r：距离变化率。
    # v_r < 0 表示双方正在接近。
    v_r = float(relative_info.get("range_rate", -float(relative_info["closing_speed"])))

    # dy/dz：目标相对拦截弹的位置。
    dy = float(relative_info["dy"])
    dz = float(relative_info["dz"])

    # dydt/dzdt：目标相对拦截弹的速度。
    dydt = float(relative_info["dydt"])
    dzdt = float(relative_info["dzdt"])

    # terminal_N：末制导比例导引系数。
    terminal_N = float(config.terminal_navigation_gain)

    # terminal_fallback：标记末制导是否退化到角误差修正。
    terminal_fallback = False

    # dqy/dqz：默认初始化，便于诊断字段统一。
    dqy = 0.0
    dqz = 0.0

    if v_r >= -EPS:
        # 如果当前没有接近趋势，则退化为角误差修正。
        terminal_fallback = True

        # tgo：使用最小时间下限，避免除零。
        tgo = float(config.minimum_tgo)

        # theta_error/psi_error：当前角误差。
        theta_error = float(relative_info["theta_error"])
        psi_error = float(relative_info["psi_error"])

        # interceptor_speed：拦截弹速度。
        interceptor_speed = max(float(interceptor_state[3]), EPS)

        # cos_theta_safe：侧向通道分母保护。
        cos_theta_safe = max(float(np.cos(theta)), 0.1)

        # ny_command：纵向角误差修正。
        ny_command = (
            cos_theta
            + interceptor_speed
            / GRAVITY
            * theta_error
            / max(tgo, float(config.minimum_tgo))
        )

        # nz_command：侧向角误差修正。
        # 根据 psi_dot = -g * nz / (V cos(theta)) 反解 nz。
        nz_command = (
            -interceptor_speed
            * cos_theta_safe
            / GRAVITY
            * psi_error
            / max(tgo, float(config.minimum_tgo))
        )

    else:
        # tgo：预计剩余交会时间。
        tgo = max(
            -distance / v_r,
            float(config.minimum_tgo),
        )

        # dqy：纵向修正视线角速度。
        dqy = float(
            -dy / (v_r * tgo**2)
            - dydt / (v_r * tgo)
        )

        # dqz：侧向修正视线角速度。
        dqz = float(
            dz / (v_r * tgo**2)
            + dzdt / (v_r * tgo)
        )

        # ny_command：纵向修正比例导引。
        ny_command = cos_theta - terminal_N * v_r * dqy / GRAVITY

        # nz_command：侧向修正比例导引。
        nz_command = -terminal_N * v_r * dqz / GRAVITY

    if not bool(config.enable_vertical_channel):
        ny_command = cos_theta

    if not bool(config.enable_lateral_channel):
        nz_command = 0.0

    # info：保存末制导各分项。
    info: Dict[str, Any] = {
        "guidance_phase": "terminal",
        "terminal_tgo": float(tgo),
        "terminal_range_rate": float(v_r),
        "terminal_dqy": float(dqy),
        "terminal_dqz": float(dqz),
        "terminal_fallback": bool(terminal_fallback),
        "guidance_raw_ny_command": float(ny_command),
        "guidance_raw_nz_command": float(nz_command),
    }

    return GuidanceCommand(
        ny_command=float(ny_command),
        nz_command=float(nz_command),
        phase="terminal",
        info=info,
    )



def select_guidance_phase(relative_info: Dict[str, float], config: Any) -> str:
    """
    判断当前制导阶段。

    参数：
        relative_info：
            相对运动信息。

            需要至少包含：
                distance：
                    当前红蓝三维距离，单位 m。

                tgo：
                    预计剩余飞行时间，单位 s。

        config：
            拦截弹配置对象。

            需要至少包含：
                terminal_distance_threshold：
                    距离小于该阈值时进入末制导。

                terminal_tgo_threshold：
                    剩余时间小于该阈值时进入末制导。

    返回：
        phase：
            当前制导阶段。

            可选：
                "midcourse"：
                    中制导阶段。

                "terminal"：
                    末制导阶段。
    """
    # distance：当前红蓝距离。
    distance = float(relative_info["distance"])

    # tgo：预计剩余飞行时间。
    tgo = float(relative_info["tgo"])

    if distance <= float(config.terminal_distance_threshold):
        return "terminal"

    if tgo <= float(config.terminal_tgo_threshold):
        return "terminal"

    return "midcourse"



def compute_mid_terminal_command(
    interceptor_state: np.ndarray,
    relative_info: Dict[str, float],
    config: Any,
    phase: Optional[str] = None,
) -> GuidanceCommand:
    """
    统一计算中制导或末制导指令。

    参数：
        interceptor_state：
            当前拦截弹状态向量。

        relative_info：
            当前相对运动信息。

        config：
            拦截弹配置对象。

        phase：
            可选制导阶段。

            如果传入：
                "midcourse"：
                    强制使用中制导。

                "terminal"：
                    强制使用末制导。

            如果为 None：
                根据距离和 tgo 自动判断阶段。

    返回：
        guidance_command：
            统一格式制导指令。
    """
    # selected_phase：如果外部没有指定阶段，则根据当前几何关系自动判断。
    selected_phase = phase
    if selected_phase is None:
        selected_phase = select_guidance_phase(relative_info=relative_info, config=config)

    if selected_phase == "midcourse":
        return compute_midcourse_command(
            interceptor_state=interceptor_state,
            relative_info=relative_info,
            config=config,
        )

    if selected_phase == "terminal":
        return compute_terminal_command(
            interceptor_state=interceptor_state,
            relative_info=relative_info,
            config=config,
        )

    raise ValueError(f"未知制导阶段：{selected_phase}")