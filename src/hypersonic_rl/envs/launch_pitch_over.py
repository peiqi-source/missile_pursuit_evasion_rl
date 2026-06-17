"""
launch_pitch_over.py

作用：
    计算发射后 pitch-over 阶段的俯仰过载指令。

设计约定：
    1. 本模块只负责生成制导指令，不推进状态；
    2. 自动驾驶仪、一阶响应、二维过载限幅和动力学积分仍由 Interceptor.step() 统一处理；
    3. pitch-over 是中末制导前的姿态管理阶段，默认不处理侧向夹击，也不加入目标机动补偿。
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from hypersonic_rl.envs.dynamics import EPS
from hypersonic_rl.envs.guidance import GuidanceCommand


def _smooth_step01(value: float) -> float:
    """
    对 [0, 1] 区间内的变量做 smooth-step 平滑。

    说明：
        这里单独实现一份，避免依赖 guidance.py 的内部私有函数。
        smooth-step 可以保证固定 20 度段向 LOS 融合时一阶连续，减少指令突变。
    """
    x = float(np.clip(value, 0.0, 1.0))
    return float(x * x * (3.0 - 2.0 * x))


def _compute_blend_weight(
    current_theta_deg: float,
    blend_start_theta_deg: float,
    blend_end_theta_deg: float,
) -> float:
    """
    根据当前弹道倾角计算 LOS 融合权重。

    约定：
        current_theta >= blend_start 时，权重为 0，参考角保持固定 pitch-over 角；
        current_theta <= blend_end 时，权重为 1，参考角由 LOS pitch angle 主导；
        中间使用 smooth-step 平滑过渡。
    """
    start_theta = float(blend_start_theta_deg)
    end_theta = float(blend_end_theta_deg)

    if start_theta <= end_theta:
        # 配置异常时退化为硬切换，避免除零或反向区间造成不可解释行为。
        return float(current_theta_deg <= end_theta)

    raw_weight = (start_theta - float(current_theta_deg)) / max(start_theta - end_theta, EPS)
    return _smooth_step01(raw_weight)


def compute_launch_pitch_over_command(
    interceptor_state: np.ndarray,
    relative_info: Dict[str, float],
    config: Any,
    elapsed_time: float,
) -> GuidanceCommand:
    """
    计算 pitch-over 阶段的纵向 / 侧向过载指令。

    参数：
        interceptor_state：
            当前拦截弹状态 [x, y, z, V, theta, psi, nx, ny, nz]。
        relative_info：
            Interceptor.compute_interceptor_relative_info() 给出的相对几何信息。
        config：
            InterceptorConfig 或具有同名字段的配置对象。
        elapsed_time：
            pitch-over 已持续时间，单位 s。

    返回：
        guidance_command：
            phase 固定为 "launch_pitch_over" 的制导指令。
    """
    # theta：当前弹道倾角，内部使用 rad，诊断同时导出 degree。
    theta = float(interceptor_state[4])
    theta_deg = float(np.degrees(theta))

    # los_theta：当前拦截弹指向目标的 LOS pitch angle。
    # 说明：
    #     compute_interceptor_relative_info() 中 desired_theta 已按 dy / horizontal_distance 计算。
    raw_los_theta = float(relative_info.get("desired_theta", 0.0))
    los_theta_min = float(np.radians(config.launch_pitch_over_los_theta_min_deg))
    los_theta_max = float(np.radians(config.launch_pitch_over_los_theta_max_deg))
    los_theta = float(np.clip(raw_los_theta, los_theta_min, los_theta_max))

    # blend_weight：从固定 20 度参考角平滑融合到 LOS pitch angle 的权重。
    blend_weight = _compute_blend_weight(
        current_theta_deg=theta_deg,
        blend_start_theta_deg=float(config.launch_pitch_over_blend_start_theta_deg),
        blend_end_theta_deg=float(config.launch_pitch_over_blend_end_theta_deg),
    )

    # reference_theta：pitch-over 参考弹道倾角。
    fixed_theta = float(np.radians(config.launch_pitch_over_fixed_theta_deg))
    reference_theta = (1.0 - blend_weight) * fixed_theta + blend_weight * los_theta
    theta_error = float(reference_theta - theta)

    # equilibrium_ny：保持当前弹道倾角的平衡项。
    equilibrium_ny = float(np.cos(theta))

    # vertical_maneuver：真正消耗机动能力的纵向额外过载。
    vertical_maneuver = float(config.launch_pitch_over_theta_gain) * theta_error
    vertical_maneuver_limit = abs(float(config.launch_pitch_over_vertical_overload_limit))
    vertical_maneuver = float(
        np.clip(vertical_maneuver, -vertical_maneuver_limit, vertical_maneuver_limit)
    )

    # ny/nz：pitch-over 阶段保持侧向安静，先完成大仰角姿态管理。
    ny_command = equilibrium_ny + vertical_maneuver
    nz_command = float(config.launch_pitch_over_lateral_overload_command)

    # ------------------------------------------------------------
    # 退出条件：当前步仍输出 pitch-over 指令，满足条件则下一步交给中制导。
    # ------------------------------------------------------------
    elapsed = float(elapsed_time)
    min_duration_met = bool(elapsed >= float(config.launch_pitch_over_min_duration))
    max_duration_met = bool(elapsed >= float(config.launch_pitch_over_max_duration))

    altitude_met = bool(float(interceptor_state[1]) >= float(config.launch_pitch_over_min_altitude))
    theta_max_met = bool(abs(theta_deg) <= float(config.launch_pitch_over_exit_theta_max_deg))
    theta_error_met = bool(
        abs(float(np.degrees(theta_error)))
        <= float(config.launch_pitch_over_exit_theta_error_deg)
    )

    if bool(config.launch_pitch_over_require_closing):
        closing_met = bool(float(relative_info.get("closing_speed", 0.0)) > 0.0)
    else:
        closing_met = True

    normal_exit_ready = bool(
        min_duration_met
        and altitude_met
        and theta_max_met
        and theta_error_met
        and closing_met
    )

    exit_ready = bool(normal_exit_ready or max_duration_met)

    if max_duration_met:
        exit_reason = "max_duration"
    elif normal_exit_ready:
        exit_reason = "ready"
    else:
        missing_conditions = []
        if not min_duration_met:
            missing_conditions.append("min_duration")
        if not altitude_met:
            missing_conditions.append("min_altitude")
        if not theta_max_met:
            missing_conditions.append("theta_max")
        if not theta_error_met:
            missing_conditions.append("theta_error")
        if not closing_met:
            missing_conditions.append("closing")
        exit_reason = ",".join(missing_conditions)

    info: Dict[str, Any] = {
        "guidance_phase": "launch_pitch_over",
        "launch_pitch_over_active": True,
        "launch_pitch_over_finished": False,
        "launch_pitch_over_elapsed_time": float(elapsed),
        "launch_pitch_over_initial_theta_deg": float(
            getattr(config, "launch_pitch_over_initial_theta_deg", np.nan)
        ),
        "launch_pitch_over_current_theta": float(theta),
        "launch_pitch_over_current_theta_deg": float(theta_deg),
        "launch_pitch_over_reference_theta": float(reference_theta),
        "launch_pitch_over_reference_theta_deg": float(np.degrees(reference_theta)),
        "launch_pitch_over_los_theta": float(los_theta),
        "launch_pitch_over_los_theta_deg": float(np.degrees(los_theta)),
        "launch_pitch_over_raw_los_theta": float(raw_los_theta),
        "launch_pitch_over_raw_los_theta_deg": float(np.degrees(raw_los_theta)),
        "launch_pitch_over_blend_weight": float(blend_weight),
        "launch_pitch_over_theta_error": float(theta_error),
        "launch_pitch_over_theta_error_deg": float(np.degrees(theta_error)),
        "launch_pitch_over_vertical_maneuver": float(vertical_maneuver),
        "launch_pitch_over_vertical_overload_limit": float(vertical_maneuver_limit),
        "launch_pitch_over_exit_ready": bool(exit_ready),
        "launch_pitch_over_exit_reason": exit_reason,
        "launch_pitch_over_min_duration_met": bool(min_duration_met),
        "launch_pitch_over_max_duration_met": bool(max_duration_met),
        "launch_pitch_over_altitude_met": bool(altitude_met),
        "launch_pitch_over_theta_max_met": bool(theta_max_met),
        "launch_pitch_over_theta_error_met": bool(theta_error_met),
        "launch_pitch_over_closing_met": bool(closing_met),
        "guidance_raw_ny_command": float(ny_command),
        "guidance_raw_nz_command": float(nz_command),
        "raw_ny_command": float(ny_command),
        "raw_nz_command": float(nz_command),
    }

    return GuidanceCommand(
        ny_command=float(ny_command),
        nz_command=float(nz_command),
        phase="launch_pitch_over",
        info=info,
    )
