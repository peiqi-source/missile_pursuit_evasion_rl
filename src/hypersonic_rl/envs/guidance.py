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
    "paper_mid_terminal",
}

def _wrap_angle(angle: float) -> float:
    """
    将角度归一化到 [-pi, pi]。

    说明：
        guidance.py 内部不要从 interceptor.py 导入 wrap_angle，
        否则容易造成循环导入。
    """
    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


def _smooth_step01(value: float) -> float:
    """
    对 [0, 1] 区间内的变量进行平滑插值。

    返回：
        smoothed：
            0 到 1 之间的平滑权重。
    """
    x = float(np.clip(value, 0.0, 1.0))
    return float(x * x * (3.0 - 2.0 * x))


def _decreasing_tgo_weight(
    tgo: float,
    start_tgo: float,
    end_tgo: float,
) -> float:
    """
    根据剩余时间生成从 0 到 1 的阶段权重。

    说明：
        当 tgo >= start_tgo 时，返回 0；
        当 tgo <= end_tgo 时，返回 1；
        中间平滑过渡。

    用途：
        用于中制导向正迎头成型、末制导交接的平滑过渡。
    """
    denominator = max(float(start_tgo) - float(end_tgo), EPS)
    raw_weight = (float(start_tgo) - float(tgo)) / denominator

    return _smooth_step01(raw_weight)


def _decreasing_value_weight(
    value: float,
    start_value: float,
    end_value: float,
) -> float:
    """
    根据单调减小的标量生成从 0 到 1 的平滑权重。

    说明：
        当 value >= start_value 时，返回 0；
        当 value <= end_value 时，返回 1；
        中间使用 smooth-step 平滑过渡。

    用途：
        可用于根据距离或剩余时间生成中末制导交接权重。
    """
    denominator = max(float(start_value) - float(end_value), EPS)
    raw_weight = (float(start_value) - float(value)) / denominator

    return _smooth_step01(raw_weight)


def _blend_angle(angle_a: float, angle_b: float, weight_b: float) -> float:
    """
    在两个角度之间做环形插值。

    参数：
        angle_a：
            起始角度。

        angle_b：
            目标角度。

        weight_b：
            目标角度权重，0 表示 angle_a，1 表示 angle_b。

    返回：
        blended_angle：
            插值后的角度，范围 [-pi, pi]。
    """
    w = float(np.clip(weight_b, 0.0, 1.0))

    sin_value = (1.0 - w) * np.sin(angle_a) + w * np.sin(angle_b)
    cos_value = (1.0 - w) * np.cos(angle_a) + w * np.cos(angle_b)

    return _wrap_angle(float(np.arctan2(sin_value, cos_value)))

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
        保存制导律计算所需的最小上下文信息。

    设计目的：
        guidance.py 只负责计算期望过载指令。
        目标机动补偿、自动驾驶仪响应、过载限幅和状态更新
        已经统一交给 Interceptor.step() 处理。

    属性：
        red_state：
            目标航迹估计向量。
            为了兼容现有环境接口，字段名仍保留为 red_state；
            在蓝方制导律内部应将其理解为 target_track，
            即由探测/跟踪系统得到的目标当前航迹估计。
            当前仿真阶段可由真值状态直接提供，后续可替换为带噪声、延迟和滤波误差的估计状态。

        interceptor_state：
            当前拦截弹状态向量。

        relative_info：
            可选相对几何信息。
            如果调用方已经计算过相对几何，可以传入这里，避免重复计算。
    """

    red_state: np.ndarray
    interceptor_state: np.ndarray
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
    # 目标机动补偿会在 Interceptor.step() 中统一加入。
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

    说明：
        对于 paper_mid_terminal，context.red_state 在接口层面保留原名，
        但在制导律内部被视为 target_track，即蓝方可用的目标航迹估计。
        当前仿真可使用目标真值作为理想航迹估计，后续可替换为带噪声和延迟的估计量。
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

        return compute_mid_terminal_command(
            interceptor_state=context.interceptor_state,
            relative_info=context.relative_info,
            config=config,
            phase=None,
        )

    if mode == "paper_mid_terminal":
        if context.relative_info is None:
            raise ValueError(
                "paper_mid_terminal 需要 context.relative_info。"
                "请先在 Interceptor 中计算相对运动信息后再传入。"
            )

        return compute_paper_mid_terminal_command(
            target_track=context.red_state,
            interceptor_state=context.interceptor_state,
            relative_info=context.relative_info,
            config=config,
            phase=None,
        )

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


def compute_paper_midcourse_command(
    target_track: np.ndarray,
    interceptor_state: np.ndarray,
    relative_info: Dict[str, float],
    config: Any,
) -> GuidanceCommand:
    """
    计算修正后的论文风格弹道成型中制导指令。

    设计原则：
        1. 不引入论文中没有出现的固定高度优势项，避免 altitude_reference = target_y + 常数 这类臆造约束；
        2. 保留论文中的“弹道成型比例导引”思想：比例导引项 + 弹道倾角/弹道偏角成型项；
        3. 纵向成型项以正的弹道倾角成型参考为主，避免远距离阶段出现先下压；
        4. 侧向成型项按两枚拦截弹相对目标所在侧自动取正负，近似对应论文中 +20° / -20° 的侧向弹道偏角成型项；
        5. 接近末制导前，中末制导由 compute_paper_mid_terminal_command() 进行平滑过渡。

    target_track：
        蓝方可用的目标航迹估计。当前仿真阶段可由目标真值状态提供，
        后续可替换为带噪声、延迟和滤波误差的目标航迹估计。
    """
    # ------------------------------------------------------------
    # 1. 基本状态量
    # ------------------------------------------------------------
    interceptor_speed = max(float(interceptor_state[3]), EPS)
    theta = float(interceptor_state[4])
    psi = float(interceptor_state[5])

    target_theta = float(target_track[4]) if len(target_track) > 4 else 0.0
    target_psi = float(target_track[5]) if len(target_track) > 5 else 0.0

    cos_theta = float(np.cos(theta))
    cos_theta_safe = max(cos_theta, 0.1)

    # ------------------------------------------------------------
    # 2. 相对几何量
    # ------------------------------------------------------------
    # dy/dz：
    #     dy > 0 表示目标高于拦截弹；
    #     dy < 0 表示目标低于拦截弹。
    dy = float(relative_info["dy"])
    dz = float(relative_info["dz"])

    closing_speed = max(float(relative_info["closing_speed"]), 0.0)
    tgo = max(float(relative_info["tgo"]), float(config.minimum_tgo))

    desired_theta = float(relative_info["desired_theta"])
    desired_psi = float(relative_info["desired_psi"])
    lambda_theta_dot = float(relative_info["lambda_theta_dot"])
    lambda_psi_dot = float(relative_info["lambda_psi_dot"])

    # ------------------------------------------------------------
    # 3. 论文风格中制导参数
    # ------------------------------------------------------------
    navigation_gain = float(
        getattr(
            config,
            "paper_midcourse_navigation_gain",
            getattr(config, "midcourse_navigation_gain", 4.0),
        )
    )

    theta_shaping_gain = float(
        getattr(
            config,
            "paper_midcourse_theta_shaping_gain",
            getattr(config, "paper_midcourse_theta_bias_gain", 0.07),
        )
    )

    psi_shaping_gain = float(
        getattr(
            config,
            "paper_midcourse_psi_shaping_gain",
            getattr(config, "paper_midcourse_psi_bias_gain", 0.09),
        )
    )

    theta_shaping_angle_rad = np.deg2rad(
        float(getattr(config, "paper_midcourse_shaping_angle_deg", 20.0))
    )

    psi_shaping_angle_rad = np.deg2rad(
        float(getattr(config, "paper_midcourse_psi_shaping_angle_deg", 20.0))
    )

    # 纵向机动分量限幅，限制的是 ny - cos(theta)，不是 ny 本身。
    vertical_maneuver_limit = float(
        getattr(config, "paper_vertical_maneuver_limit", 3.5)
    )

    # 侧向成型分量限幅，仅限制额外侧向成型项，不限制 PN 主项。
    lateral_shaping_limit = float(
        getattr(config, "paper_lateral_shaping_limit", 4.0)
    )

    # 正迎头成型权重：远距离主要按弹道偏角成型，中后段逐渐转向目标速度反方向。
    headon_start_tgo = float(getattr(config, "paper_headon_start_tgo", 35.0))
    headon_end_tgo = float(getattr(config, "paper_headon_end_tgo", 10.0))
    headon_weight = _decreasing_tgo_weight(
        tgo=tgo,
        start_tgo=headon_start_tgo,
        end_tgo=headon_end_tgo,
    )

    # 中末交接权重：接近末制导时，纵向中制导参考逐渐贴近目标视线倾角。
    handover_start_tgo = float(getattr(config, "paper_handover_start_tgo", 18.0))
    handover_end_tgo = float(getattr(config, "terminal_tgo_threshold", 4.0))
    handover_weight = _decreasing_tgo_weight(
        tgo=tgo,
        start_tgo=handover_start_tgo,
        end_tgo=handover_end_tgo,
    )

    # ------------------------------------------------------------
    # 4. 纵向弹道倾角成型参考
    # ------------------------------------------------------------
    # ------------------------------------------------------------
    # 4. 纵向弹道倾角自适应成型参考
    # ------------------------------------------------------------
    # 论文表中给出中制导弹道倾角成型项为 20°。
    # 这里不将 20° 固定解释为 theta = target_theta + 20°，
    # 而是把它作为“最大弹道倾角成型幅值”。
    #
    # 自适应逻辑：
    #   1. 目标高于拦截弹时，成型偏置为正，拦截弹上修；
    #   2. 目标低于拦截弹时，成型偏置为负，拦截弹下修；
    #   3. 双方高度接近时，保留较弱正成型，贴近论文基准场景；
    #   4. 接近末制导时，该成型偏置逐渐消失，回到目标视线角。
    altitude_transition_band = max(
        float(getattr(config, "paper_altitude_transition_band", 5000.0)),
        EPS,
    )

    neutral_climb_bias_ratio = float(
        getattr(config, "paper_neutral_climb_bias_ratio", 0.30)
    )

    downward_bias_ratio = float(
        getattr(config, "paper_downward_bias_ratio", 0.70)
    )

    # height_ratio：
    #     dy / altitude_transition_band 的归一化高度关系。
    #     +1 表示目标明显高于拦截弹；
    #     -1 表示目标明显低于拦截弹；
    #      0 表示双方高度接近。
    height_ratio = float(
        np.clip(
            dy / altitude_transition_band,
            -1.0,
            1.0,
        )
    )

    # upward_or_downward_ratio：
    #     目标在上方时允许完整正向成型；
    #     目标在下方时只使用 downward_bias_ratio 控制下压强度，
    #     避免拦截弹高于目标时过度俯冲。
    if height_ratio >= 0.0:
        upward_or_downward_ratio = height_ratio
    else:
        upward_or_downward_ratio = downward_bias_ratio * height_ratio

    # neutral_ratio：
    #     当双方高度接近时，保留一个较弱的正成型项。
    #     这对应论文同高度初始场景下仍然存在弹道倾角成型项的思想。
    neutral_ratio = neutral_climb_bias_ratio * (1.0 - abs(height_ratio))

    # adaptive_theta_shape_ratio：
    #     最终自适应成型比例。
    #     低于目标时趋近 +1；
    #     高于目标时趋近 -downward_bias_ratio；
    #     同高度时约为 neutral_climb_bias_ratio。
    adaptive_theta_shape_ratio = float(
        np.clip(
            upward_or_downward_ratio + neutral_ratio,
            -downward_bias_ratio,
            1.0,
        )
    )

    adaptive_theta_shape_bias = float(
        theta_shaping_angle_rad * adaptive_theta_shape_ratio
    )

    # handover_theta_shape_bias：
    #     接近末制导时逐渐消失，保证中末制导平滑交接。
    handover_theta_shape_bias = float(
        (1.0 - handover_weight) * adaptive_theta_shape_bias
    )

    # theta_reference：
    #     以当前目标视线倾角 desired_theta 为几何基准，
    #     叠加随高度关系自适应变化的弹道成型偏置。
    theta_reference = float(
        desired_theta + handover_theta_shape_bias
    )

    theta_reference_error = _wrap_angle(theta_reference - theta)

    # ------------------------------------------------------------
    # 5.1 侧向成型符号
    # ------------------------------------------------------------
    # 优先使用 InterceptorFleet.reset() 为每枚弹固定的初始侧向符号。
    # 这样可以避免 dz 穿越 0 时，+20° / -20° 突然翻转。
    #
    # 如果没有固定符号，则退化为旧的 dynamic_dz 逻辑；
    # 但 dz=0 时使用 paper_centerline_side_sign，默认 0，不再强行 +20°。
    fixed_side_sign = getattr(config, "paper_fixed_side_sign", None)
    side_sign_mode = str(getattr(config, "paper_side_sign_mode", "fixed_initial"))
    centerline_side_sign = float(getattr(config, "paper_centerline_side_sign", 0.0))

    if fixed_side_sign is not None and side_sign_mode == "fixed_initial":
        fixed_side_sign = float(fixed_side_sign)

        if abs(fixed_side_sign) > EPS:
            side_sign = float(np.sign(fixed_side_sign))
        else:
            side_sign = 0.0

        side_sign_source = "fixed_initial"

    elif side_sign_mode == "none":
        side_sign = 0.0
        side_sign_source = "none"

    else:
        # 兼容旧逻辑：没有固定符号时，才按当前 dz 判断。
        if abs(dz) > EPS:
            side_sign = float(np.sign(dz))
            side_sign_source = "dynamic_dz"
        else:
            side_sign = centerline_side_sign
            side_sign_source = "centerline"

    paper_psi_reference_biased = _wrap_angle(
        desired_psi + side_sign * psi_shaping_angle_rad
    )

    # 近似正迎头参考：目标航向的反方向。
    headon_psi_reference = _wrap_angle(target_psi + np.pi)

    # 中段逐渐由弹道偏角成型参考过渡到正迎头参考。
    psi_reference = _blend_angle(
        paper_psi_reference_biased,
        headon_psi_reference,
        headon_weight,
    )

    psi_reference_error = _wrap_angle(psi_reference - psi)

    # ------------------------------------------------------------
    # 6. 比例导引项
    # ------------------------------------------------------------
    paper_pn_y = (
        navigation_gain
        * closing_speed
        * lambda_theta_dot
        / GRAVITY
    )

    # 当前动力学中 psi_dot = -g * nz / (V cos(theta))，所以侧向 PN 项带负号。
    paper_pn_z = (
        -navigation_gain
        * closing_speed
        * lambda_psi_dot
        / GRAVITY
    )

    # ------------------------------------------------------------
    # 7. 弹道成型项
    # ------------------------------------------------------------
    paper_theta_shaping_y = (
        interceptor_speed
        / GRAVITY
        * theta_shaping_gain
        * theta_reference_error
        / tgo
    )

    paper_psi_shaping_z = (
        -interceptor_speed
        * cos_theta_safe
        / GRAVITY
        * psi_shaping_gain
        * psi_reference_error
        / tgo
    )

    # ------------------------------------------------------------
    # 8. 分项限幅与通道开关
    # ------------------------------------------------------------
    vertical_maneuver = float(
        np.clip(
            paper_pn_y + paper_theta_shaping_y,
            -vertical_maneuver_limit,
            vertical_maneuver_limit,
        )
    )

    lateral_shaping = float(
        np.clip(
            paper_psi_shaping_z,
            -lateral_shaping_limit,
            lateral_shaping_limit,
        )
    )

    ny_command = cos_theta + vertical_maneuver
    nz_command = paper_pn_z + lateral_shaping

    if not bool(config.enable_vertical_channel):
        ny_command = cos_theta

    if not bool(config.enable_lateral_channel):
        nz_command = 0.0

    # ------------------------------------------------------------
    # 9. 阶段标签与诊断信息
    # ------------------------------------------------------------
    if handover_weight > 0.0:
        guidance_phase = "paper_midcourse_handover"
    elif headon_weight > 0.0:
        guidance_phase = "paper_midcourse_headon"
    else:
        guidance_phase = "paper_midcourse_shaping"

    info: Dict[str, Any] = {
        **relative_info,
        "guidance_phase": guidance_phase,
        "paper_midcourse_tgo": float(tgo),
        "paper_midcourse_navigation_gain": float(navigation_gain),
        "paper_midcourse_theta_shaping_gain": float(theta_shaping_gain),
        "paper_midcourse_psi_shaping_gain": float(psi_shaping_gain),
        "paper_midcourse_theta_shaping_angle_rad": float(theta_shaping_angle_rad),
        "paper_midcourse_psi_shaping_angle_rad": float(psi_shaping_angle_rad),
        "paper_midcourse_headon_weight": float(headon_weight),
        "paper_midcourse_handover_weight": float(handover_weight),
                "paper_midcourse_target_theta": float(target_theta),
        "paper_midcourse_target_psi": float(target_psi),
        "paper_midcourse_desired_theta": float(desired_theta),
        "paper_midcourse_desired_psi": float(desired_psi),

        "paper_midcourse_dy": float(dy),
        "paper_midcourse_altitude_transition_band": float(altitude_transition_band),
        "paper_midcourse_height_ratio": float(height_ratio),
        "paper_midcourse_neutral_climb_bias_ratio": float(neutral_climb_bias_ratio),
        "paper_midcourse_downward_bias_ratio": float(downward_bias_ratio),
        "paper_midcourse_adaptive_theta_shape_ratio": float(adaptive_theta_shape_ratio),
        "paper_midcourse_adaptive_theta_shape_bias": float(adaptive_theta_shape_bias),
        "paper_midcourse_handover_theta_shape_bias": float(handover_theta_shape_bias),

        "paper_midcourse_theta_reference": float(theta_reference),
        "paper_midcourse_theta_reference_error": float(theta_reference_error),
        "paper_midcourse_psi_reference": float(psi_reference),
        "paper_midcourse_headon_psi_reference": float(headon_psi_reference),
        "paper_midcourse_psi_reference_error": float(psi_reference_error),
        "paper_midcourse_side_sign": float(side_sign),
        "paper_midcourse_fixed_side_sign": (
            float(fixed_side_sign) if fixed_side_sign is not None else np.nan
        ),
        "paper_midcourse_centerline_side_sign": float(centerline_side_sign),
        "paper_midcourse_pn_y": float(paper_pn_y),
        "paper_midcourse_theta_shaping_y": float(paper_theta_shaping_y),
        "paper_midcourse_vertical_maneuver": float(vertical_maneuver),
        "paper_midcourse_pn_z": float(paper_pn_z),
        "paper_midcourse_psi_shaping_z": float(paper_psi_shaping_z),
        "paper_midcourse_lateral_shaping": float(lateral_shaping),
        "guidance_raw_ny_command": float(ny_command),
        "guidance_raw_nz_command": float(nz_command),
        "raw_ny_command": float(ny_command),
        "raw_nz_command": float(nz_command),
    }

    return GuidanceCommand(
        ny_command=float(ny_command),
        nz_command=float(nz_command),
        phase=guidance_phase,
        info=info,
    )

def compute_paper_terminal_command(
    interceptor_state: np.ndarray,
    relative_info: Dict[str, float],
    config: Any,
) -> GuidanceCommand:
    """
    计算论文风格末制导指令。

    说明：
        该函数对应 paper_mid_terminal 模式下的末制导阶段。

        与普通 terminal 的区别：
            1. 本函数不再复用 compute_terminal_command()；
            2. 侧向通道加入 cos(theta) 修正，更贴近论文/legacy 中
               nzc = N * dr * dqz * cos(theta) / g 的形式；
            3. 所有诊断字段使用 paper_terminal_* 前缀，便于和普通 terminal 区分；
            4. 目标机动补偿仍然不在本函数中处理，而由 Interceptor.step()
               统一加入，保持 guidance.py 只输出期望过载指令。

    返回：
        guidance_command：
            论文风格末制导期望过载指令。
    """
    # theta：拦截弹当前航迹倾角。
    theta = float(interceptor_state[4])

    # cos_theta：纵向平衡项。
    cos_theta = float(np.cos(theta))

    # cos_theta_safe：侧向通道保护项，避免接近垂直飞行时数值过激。
    cos_theta_safe = max(float(np.cos(theta)), 0.1)

    # distance：当前红蓝三维距离。
    distance = max(float(relative_info["distance"]), EPS)

    # range_rate：距离变化率。
    # range_rate < 0 表示正在接近。
    range_rate = float(
        relative_info.get("range_rate", -float(relative_info["closing_speed"]))
    )

    # dy/dz：目标相对拦截弹的位置分量。
    dy = float(relative_info["dy"])
    dz = float(relative_info["dz"])

    # dydt/dzdt：目标相对拦截弹的速度分量。
    dydt = float(relative_info["dydt"])
    dzdt = float(relative_info["dzdt"])

    # terminal_gain：论文风格末制导比例系数。
    terminal_gain = float(
        getattr(config, "paper_terminal_navigation_gain", config.terminal_navigation_gain)
    )

    # paper_terminal_fallback：标记是否退化到角误差修正。
    paper_terminal_fallback = False

    # dqy/dqz：默认初始化，保证 info 字段稳定。
    dqy = 0.0
    dqz = 0.0

    if range_rate >= -EPS:
        # 没有接近趋势时，无法稳定估计 tgo 和 dqy/dqz。
        # 此时退化为角误差修正，避免末制导指令发散。
        paper_terminal_fallback = True

        # tgo：使用最小时间下限。
        tgo = float(config.minimum_tgo)

        # theta_error/psi_error：当前角误差。
        theta_error = float(relative_info["theta_error"])
        psi_error = float(relative_info["psi_error"])

        # interceptor_speed：拦截弹速度。
        interceptor_speed = max(float(interceptor_state[3]), EPS)

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
            -distance / range_rate,
            float(config.minimum_tgo),
        )

        # dqy：纵向修正视线角速度。
        dqy = float(
            -dy / (range_rate * tgo**2)
            - dydt / (range_rate * tgo)
        )

        # dqz：侧向修正视线角速度。
        dqz = float(
            dz / (range_rate * tgo**2)
            + dzdt / (range_rate * tgo)
        )

        # paper_terminal_y：
        #     纵向论文风格末制导项。
        #     与普通 terminal 保持相同主结构。
        paper_terminal_y = -terminal_gain * range_rate * dqy / GRAVITY

        # paper_terminal_z：
        #     侧向论文风格末制导项。
        #     相比普通 terminal，多乘 cos(theta)，对应旧公式中的侧向通道修正。
        if bool(getattr(config, "paper_terminal_use_cos_theta", True)):
            lateral_cos_factor = cos_theta_safe
        else:
            lateral_cos_factor = 1.0

        paper_terminal_z = (
                -terminal_gain
                * range_rate
                * lateral_cos_factor
                * dqz
                / GRAVITY
        )

        # ny_command/nz_command：论文风格末制导指令。
        ny_command = cos_theta + paper_terminal_y
        nz_command = paper_terminal_z

    # 通道开关。
    if not bool(config.enable_vertical_channel):
        ny_command = cos_theta

    if not bool(config.enable_lateral_channel):
        nz_command = 0.0

    # info：保存 paper terminal 独立诊断字段。
    info: Dict[str, Any] = {
        **relative_info,
        "guidance_phase": "paper_terminal",
        "paper_terminal_tgo": float(tgo),
        "paper_terminal_range_rate": float(range_rate),
        "paper_terminal_dqy": float(dqy),
        "paper_terminal_dqz": float(dqz),
        "paper_terminal_gain": float(terminal_gain),
        "paper_terminal_cos_theta": float(cos_theta_safe),
        "paper_terminal_fallback": bool(paper_terminal_fallback),
        "guidance_raw_ny_command": float(ny_command),
        "guidance_raw_nz_command": float(nz_command),
        "raw_ny_command": float(ny_command),
        "raw_nz_command": float(nz_command),
        "paper_terminal_use_cos_theta": bool(
            getattr(config, "paper_terminal_use_cos_theta", True)
        ),
    }

    return GuidanceCommand(
        ny_command=float(ny_command),
        nz_command=float(nz_command),
        phase="paper_terminal",
        info=info,
    )



def compute_paper_mid_terminal_command(
    target_track: np.ndarray,
    interceptor_state: np.ndarray,
    relative_info: Dict[str, float],
    config: Any,
    phase: Optional[str] = None,
) -> GuidanceCommand:
    """
    统一计算论文风格中制导、平滑交接或末制导指令。

    说明：
        论文采用中段制导 + 末段制导两阶段结构，并要求中末交接形成正向拦截态势。
        工程实现中如果直接硬切换，容易造成 ny/nz 指令突变。
        因此这里在 terminal_tgo_threshold 附近加入平滑交接带：
            far：纯 paper_midcourse；
            transition：paper_midcourse 与 paper_terminal 按权重融合；
            near：纯 paper_terminal。
    """
    selected_phase = phase

    if selected_phase == "midcourse":
        return compute_paper_midcourse_command(
            target_track=target_track,
            interceptor_state=interceptor_state,
            relative_info=relative_info,
            config=config,
        )

    if selected_phase == "terminal":
        return compute_paper_terminal_command(
            interceptor_state=interceptor_state,
            relative_info=relative_info,
            config=config,
        )

    if selected_phase is not None:
        raise ValueError(f"未知论文风格制导阶段：{selected_phase}")

    # ------------------------------------------------------------
    # 1. 计算中制导与末制导候选指令
    # ------------------------------------------------------------
    midcourse_command = compute_paper_midcourse_command(
        target_track=target_track,
        interceptor_state=interceptor_state,
        relative_info=relative_info,
        config=config,
    )

    terminal_command = compute_paper_terminal_command(
        interceptor_state=interceptor_state,
        relative_info=relative_info,
        config=config,
    )

    # ------------------------------------------------------------
    # 2. 计算中末交接融合权重
    # ------------------------------------------------------------
    distance = float(relative_info["distance"])
    tgo = float(relative_info["tgo"])

    terminal_tgo_threshold = float(getattr(config, "terminal_tgo_threshold", 4.0))
    terminal_distance_threshold = float(getattr(config, "terminal_distance_threshold", 10000.0))

    blend_start_tgo = float(
        getattr(
            config,
            "paper_terminal_blend_start_tgo",
            max(terminal_tgo_threshold + 4.0, terminal_tgo_threshold * 2.0),
        )
    )
    blend_end_tgo = float(
        getattr(config, "paper_terminal_blend_end_tgo", terminal_tgo_threshold)
    )

    blend_start_distance = float(
        getattr(
            config,
            "paper_terminal_blend_start_distance",
            max(terminal_distance_threshold * 1.5, terminal_distance_threshold + 5000.0),
        )
    )
    blend_end_distance = float(
        getattr(config, "paper_terminal_blend_end_distance", terminal_distance_threshold)
    )

    tgo_weight = _decreasing_value_weight(
        value=tgo,
        start_value=blend_start_tgo,
        end_value=blend_end_tgo,
    )

    distance_weight = _decreasing_value_weight(
        value=distance,
        start_value=blend_start_distance,
        end_value=blend_end_distance,
    )

    terminal_blend_weight = float(np.clip(max(tgo_weight, distance_weight), 0.0, 1.0))

    # ------------------------------------------------------------
    # 3. 根据权重返回对应阶段
    # ------------------------------------------------------------
    if terminal_blend_weight <= 0.0:
        info = dict(midcourse_command.info)
        info["paper_terminal_blend_weight"] = 0.0
        info["paper_terminal_tgo_blend_weight"] = float(tgo_weight)
        info["paper_terminal_distance_blend_weight"] = float(distance_weight)
        info["paper_terminal_blend_start_tgo"] = float(blend_start_tgo)
        info["paper_terminal_blend_end_tgo"] = float(blend_end_tgo)
        info["paper_terminal_blend_start_distance"] = float(blend_start_distance)
        info["paper_terminal_blend_end_distance"] = float(blend_end_distance)
        return GuidanceCommand(
            ny_command=float(midcourse_command.ny_command),
            nz_command=float(midcourse_command.nz_command),
            phase=str(midcourse_command.phase),
            info=info,
        )

    if terminal_blend_weight >= 1.0:
        info = dict(terminal_command.info)
        info["paper_terminal_blend_weight"] = 1.0
        info["paper_terminal_tgo_blend_weight"] = float(tgo_weight)
        info["paper_terminal_distance_blend_weight"] = float(distance_weight)
        info["paper_terminal_blend_start_tgo"] = float(blend_start_tgo)
        info["paper_terminal_blend_end_tgo"] = float(blend_end_tgo)
        info["paper_terminal_blend_start_distance"] = float(blend_start_distance)
        info["paper_terminal_blend_end_distance"] = float(blend_end_distance)
        return GuidanceCommand(
            ny_command=float(terminal_command.ny_command),
            nz_command=float(terminal_command.nz_command),
            phase="paper_terminal",
            info=info,
        )

    # ------------------------------------------------------------
    # 4. 平滑融合中制导与末制导指令
    # ------------------------------------------------------------
    w = terminal_blend_weight
    ny_command = (1.0 - w) * float(midcourse_command.ny_command) + w * float(terminal_command.ny_command)
    nz_command = (1.0 - w) * float(midcourse_command.nz_command) + w * float(terminal_command.nz_command)

    info: Dict[str, Any] = dict(midcourse_command.info)

    # 保留末制导诊断字段，避免覆盖中制导主诊断。
    for key, value in terminal_command.info.items():
        if key.startswith("paper_terminal_"):
            info[key] = value

    info.update(
        {
            "guidance_phase": "paper_mid_terminal_blend",
            "paper_midcourse_raw_ny_command": float(midcourse_command.ny_command),
            "paper_midcourse_raw_nz_command": float(midcourse_command.nz_command),
            "paper_terminal_raw_ny_command": float(terminal_command.ny_command),
            "paper_terminal_raw_nz_command": float(terminal_command.nz_command),
            "paper_terminal_blend_weight": float(w),
            "paper_terminal_tgo_blend_weight": float(tgo_weight),
            "paper_terminal_distance_blend_weight": float(distance_weight),
            "paper_terminal_blend_start_tgo": float(blend_start_tgo),
            "paper_terminal_blend_end_tgo": float(blend_end_tgo),
            "paper_terminal_blend_start_distance": float(blend_start_distance),
            "paper_terminal_blend_end_distance": float(blend_end_distance),
            "guidance_raw_ny_command": float(ny_command),
            "guidance_raw_nz_command": float(nz_command),
            "raw_ny_command": float(ny_command),
            "raw_nz_command": float(nz_command),
        }
    )

    return GuidanceCommand(
        ny_command=float(ny_command),
        nz_command=float(nz_command),
        phase="paper_mid_terminal_blend",
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