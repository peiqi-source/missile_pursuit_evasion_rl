"""
pursue_escape_env.py

作用：
    实现高超声速飞行器一对二突防强化学习环境。

当前任务定义：
    红方：
        高超声速飞行器，由 SAC 智能体输出 1 维横向过载指令。
    蓝方：
        一枚或两枚拦截弹，由工程制导律闭环控制。

默认论文接口：
    observation = [r1, dr1, q1, dq1, r2, dr2, q2, dq2, rHT, nHz]
    action      = [nzc_h]
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from hypersonic_rl.envs.autopilot import FirstOrderAutopilot
from hypersonic_rl.envs.dynamics import (
    EPS,
    build_velocity_vector,
    compute_relative_geometry,
    degrees_to_radians,
    update_point_mass_state,
)
from hypersonic_rl.envs.guidance import (
    ProportionalNavigationConfig,
    SUPPORTED_GUIDANCE_MODES,
)
from hypersonic_rl.envs.interceptor import InterceptorConfig
from hypersonic_rl.envs.interceptor_fleet import InterceptorFleet, compute_segment_closest_distance
from hypersonic_rl.envs.reward import RewardConfig, calculate_end_to_end_reward


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
    # wrapped_angle：使用模运算消除 2pi 周期差异。
    wrapped_angle = (float(angle) + np.pi) % (2.0 * np.pi) - np.pi

    return float(wrapped_angle)


@dataclass
class PursueEscapeEnvConfig:
    """
    PursueEscapeEnvConfig

    作用：
        作为 PursueEscapeEnv 的环境总配置入口。

    配置链路：
        YAML 配置文件
            -> load_config_from_project()
            -> build_dataclass_config(..., PursueEscapeEnvConfig)
            -> PursueEscapeEnv(config)
            -> ProportionalNavigationConfig / RewardConfig / InterceptorConfig / InterceptorFleet

    关键约定：
        1. YAML 中的字段名必须与本 dataclass 字段名完全一致；
        2. 凡是希望通过 YAML 修改并影响环境的变量，都应该先进入本配置类；
        3. 本配置类只管理环境相关参数，不管理 SAC 学习率、batch_size、replay buffer 等算法训练参数；
        4. scenario_profile 决定哪些初始态势字段真正生效；
        5. interceptor_ability_profile 若不是 custom，会覆盖显式过载上限。
    """

    # ============================================================
    # 1. 环境接口与任务规模
    # ============================================================

    # 红方动作维度固定为 1，对应横向过载指令 nzc_h。
    # 该值表示智能体动作的最大绝对值，单位 g。
    nzc_h_max: float = 2.0

    # 默认观测模式。
    # 当前仅支持 thesis_end_to_end_10d：
    #     [r1, dr1, q1, dq1, r2, dr2, q2, dq2, rHT, nHz]
    observation_mode: str = "thesis_end_to_end_10d"

    # 拦截弹数量。
    # 当前环境支持 1 或 2；正式一对二突防训练使用 2。
    interceptor_count: int = 2

    # 蓝方制导模式。
    # 可选：
    #     source_pn：
    #         单阶段修正比例导引 baseline，主要用于快速闭环验证；
    #     mid_terminal_interceptor：
    #         当前工程化中末制导 baseline；
    #     paper_mid_terminal：
    #         论文风格弹道成型中制导 + 末制导 baseline。
    guidance_mode: str = "mid_terminal_interceptor"

    # 初始态势 profile。
    # 可选：
    #     paper_200km_end_to_end：
    #         论文默认 200 km 一对二训练态势，使用 paper_interceptor_* 参数；
    #     custom：
    #         使用 interceptor_initial_*_min/max 范围生成拦截弹初始位置；
    #     manual_pair：
    #         手动指定两枚拦截弹初始状态，适合固定对照实验和可视化诊断。
    scenario_profile: str = "paper_200km_end_to_end"

    # ============================================================
    # 2. 仿真时间、动力学积分与自动驾驶仪
    # ============================================================

    # 仿真步长，单位 s。
    # 注意：杀伤半径只有 5 m 时，dt 对命中 / 错过判断很敏感；正式实验建议使用 0.01。
    dt: float = 0.01

    # 单回合最大仿真时间，单位 s。
    # max_steps = ceil(t / dt)。
    t: float = 80.0

    # 红方一阶自动驾驶仪时间常数，单位 s。
    tau_h: float = 0.5

    # 拦截弹一阶自动驾驶仪时间常数，单位 s。
    tau_i: float = 0.5

    # 质点运动学积分方法。
    # 可选：
    #     semi_implicit_euler：
    #         半隐式欧拉 / 辛欧拉，速度更新后再更新位置，适合快速调试；
    #     explicit_euler：
    #         显式欧拉，仅用于对照；
    #     rk4：
    #         四阶龙格库达法，数值精度更高，正式实验建议使用。
    dynamics_integration_mode: str = "rk4"

    # 红方自动驾驶仪输出变化率限制，单位 g/s。
    # None 表示不额外限速，只使用一阶惯性响应。
    red_autopilot_rate_limit: Optional[float] = None

    # 拦截弹自动驾驶仪输出变化率限制，单位 g/s。
    # None 表示不额外限速，只使用一阶惯性响应。
    interceptor_autopilot_rate_limit: Optional[float] = None

    # ============================================================
    # 3. 速度、杀伤半径与基础物理量
    # ============================================================

    # Mach 到 m/s 的工程换算声速。
    sound_speed: float = 340.0

    # 红方默认速度，单位 Mach。
    red_mach: float = 6.0

    # 拦截弹默认速度，单位 Mach。
    interceptor_mach: float = 4.0

    # 课程训练可选红方速度随机化范围，单位 Mach。
    # 两者同时不为 None 时，每回合从 [red_mach_min, red_mach_max] 采样。
    red_mach_min: Optional[float] = None
    red_mach_max: Optional[float] = None

    # 课程训练可选拦截弹速度随机化范围，单位 Mach。
    # 两者同时不为 None 时，每回合从 [interceptor_mach_min, interceptor_mach_max] 采样。
    interceptor_mach_min: Optional[float] = None
    interceptor_mach_max: Optional[float] = None

    # 杀伤半径，单位 m。
    # 当任一拦截弹与红方的最近距离小于该值时，判定红方被拦截。
    kill_radius: float = 5.0

    # ============================================================
    # 4. 红方初始状态与目标位置
    # ============================================================

    # 红方初始位置，单位 m。
    red_initial_x: float = 0.0
    red_initial_y: float = 25000.0
    red_initial_z: float = 0.0

    # 红方初始弹道倾角 theta，单位 degree。
    # 0 表示水平飞行。
    red_initial_theta_deg: float = 0.0

    # 红方初始航向角 psi，单位 degree。
    # 0 表示沿 +x 方向飞行。
    red_initial_psi_deg: float = 0.0

    # 红方初始航向角额外随机拉偏范围，单位 degree。
    # 两者相等时表示不随机拉偏。
    red_initial_psi_delta_min_deg: float = 0.0
    red_initial_psi_delta_max_deg: float = 0.0

    # 预设打击目标位置，单位 m。
    # 当前主要用于计算 rHT 和端到端过程奖励。
    target_initial_x: float = 220000.0
    target_initial_y: float = 25000.0
    target_initial_z: float = 0.0

    # ============================================================
    # 5. 初始随机化总开关
    # ============================================================

    # 是否启用论文第 5 章训练随机化。
    # True：
    #     在 profile 给出的基础初始态势上加入位置 / 航向 / 弹道倾角扰动；
    # False：
    #     使用固定初始态势，适合闭环检查和可视化对照。
    initial_randomization_enabled: bool = True

    # 拦截弹初始位置随机扰动范围，单位 m。
    # paper_200km_end_to_end 和 manual_pair 下，表示在基础位置周围加均匀扰动。
    interceptor_position_randomization_m: float = 3000.0

    # 是否允许拦截弹初始高度 y 方向也参与随机扰动。
    # False：
    #     只扰动 x-z 平面，更接近当前论文训练设定；
    # True：
    #     x-y-z 三个方向都扰动，适合后续鲁棒训练。
    randomize_interceptor_y: bool = False

    # 拦截弹初始航向角随机扰动范围，单位 degree。
    interceptor_heading_randomization_deg: float = 3.0

    # 拦截弹初始弹道倾角随机扰动范围，单位 degree。
    interceptor_theta_randomization_deg: float = 0.0

    # ============================================================
    # 6. paper_200km_end_to_end profile 参数
    # ============================================================

    # paper profile：拦截弹相对红方的前向距离，单位 m。
    # 计算方式：
    #     interceptor_x = red_initial_x + paper_interceptor_x_distance
    # 例：
    #     red_initial_x=0 且 paper_interceptor_x_distance=200000 表示拦截弹初始 x=200 km。
    paper_interceptor_x_distance: float = 200000.0

    # paper profile：双弹侧向夹击偏置绝对值，单位 m。
    # interceptor_count=2 时，两枚弹分别位于 z=-offset 和 z=+offset。
    # interceptor_count=1 时，一般使用中心线或单侧位置，具体由环境构造逻辑决定。
    paper_interceptor_lateral_offset: float = 10000.0

    # paper profile：拦截弹相对红方的高度偏置，单位 m。
    # 计算方式：
    #     interceptor_y = red_initial_y + paper_interceptor_y_offset_from_red
    # 例：
    #     -2000 表示拦截弹低于红方 2 km；
    #      0    表示同高度；
    #     +2000 表示拦截弹高于红方 2 km。
    paper_interceptor_y_offset_from_red: float = 0.0

    # paper profile：拦截弹基础初始弹道倾角，单位 degree。
    # 说明：
    #     该值只控制拦截弹初始速度方向；
    paper_interceptor_theta_deg: float = 0.0

    # ============================================================
    # 7. custom profile 参数
    # ============================================================

    # custom profile 下拦截弹初始位置随机范围，单位 m。
    # 只有 scenario_profile="custom" 时，这组参数才控制拦截弹初始位置。
    interceptor_initial_x_min: float = 200000.0
    interceptor_initial_x_max: float = 200000.0
    interceptor_initial_y_min: float = 23000.0
    interceptor_initial_y_max: float = 23000.0
    interceptor_initial_z_min: float = 0.0
    interceptor_initial_z_max: float = 0.0

    # custom profile 下拦截弹基础初始弹道倾角，单位 degree。
    interceptor_initial_theta_deg: float = 0.0

    # custom profile 下拦截弹初始航向角采样范围，单位 degree。
    interceptor_initial_heading_min_deg: float = 180.0
    interceptor_initial_heading_max_deg: float = 180.0

    # ============================================================
    # 8. manual_pair profile 参数
    # ============================================================

    # manual_pair profile 说明：
    #     该模式适合固定对照实验、两弹初始条件诊断和论文图复现；
    #     如果 initial_randomization_enabled=True，则会在手动初始状态周围继续加入随机扰动；
    #     如果 interceptor_count=1，通常只使用第一枚拦截弹配置。

    # 第一枚拦截弹初始位置，单位 m。
    manual_interceptor_1_x: float = 200000.0
    manual_interceptor_1_y: float = 23000.0
    manual_interceptor_1_z: float = -10000.0

    # 第一枚拦截弹初始弹道倾角和航向角，单位 degree。
    manual_interceptor_1_theta_deg: float = 0.0
    manual_interceptor_1_heading_deg: float = 180.0

    # 第二枚拦截弹初始位置，单位 m。
    manual_interceptor_2_x: float = 200000.0
    manual_interceptor_2_y: float = 23000.0
    manual_interceptor_2_z: float = 10000.0

    # 第二枚拦截弹初始弹道倾角和航向角，单位 degree。
    manual_interceptor_2_theta_deg: float = 0.0
    manual_interceptor_2_heading_deg: float = 180.0

    # ============================================================
    # 9. 蓝方能力档位与 source_pn 配置
    # ============================================================

    # 蓝方能力档位。
    # 可选：
    #     custom：
    #         不覆盖过载上限，直接使用 source_pn_max_overload 和 interceptor_max_overload；
    #     weak：
    #         自动覆盖为 6 g；
    #     paper：
    #         自动覆盖为 8 g；
    #     strong：
    #         自动覆盖为 12 g。
    interceptor_ability_profile: str = "custom"

    # source_pn 模式下拦截弹最大机动过载，单位 g。
    # 注意：当 interceptor_ability_profile 不是 custom 时，该值会被能力档位覆盖。
    source_pn_max_overload: float = 8.0

    # source_pn 比例导引系数。
    N: float = 4.0

    # source_pn 对红方横向机动的前馈补偿系数。
    source_pn_compensation_gain: float = 0.5

    # ============================================================
    # 10. 发射后 pitch-over 阶段配置
    # ============================================================

    # 是否启用发射后 pitch-over 阶段。
    enable_launch_pitch_over: bool = False

    # pitch-over 适用的制导模式，使用逗号分隔。
    launch_pitch_over_guidance_modes: str = "mid_terminal_interceptor,paper_mid_terminal"

    # pitch-over 自动激活阈值，单位 degree。
    launch_pitch_over_activation_theta_deg: float = 45.0

    # pitch-over 前段固定参考弹道倾角，单位 degree。
    launch_pitch_over_fixed_theta_deg: float = 20.0

    # pitch-over 后段 LOS 融合开始角，单位 degree。
    launch_pitch_over_blend_start_theta_deg: float = 30.0

    # pitch-over 后段 LOS 融合结束角，单位 degree。
    launch_pitch_over_blend_end_theta_deg: float = 20.0

    # LOS pitch angle 限幅，单位 degree。
    launch_pitch_over_los_theta_min_deg: float = -5.0
    launch_pitch_over_los_theta_max_deg: float = 25.0

    # pitch-over 纵向控制增益，单位约为 g/rad。
    launch_pitch_over_theta_gain: float = 3.0

    # pitch-over 纵向额外机动限幅，单位 g。
    launch_pitch_over_vertical_overload_limit: float = 5.0

    # pitch-over 阶段侧向控制指令，单位 g。
    launch_pitch_over_lateral_overload_command: float = 0.0

    # pitch-over 最短 / 最长持续时间，单位 s。
    launch_pitch_over_min_duration: float = 2.0
    launch_pitch_over_max_duration: float = 12.0

    # pitch-over 结束所需最低高度，单位 m。
    launch_pitch_over_min_altitude: float = 1000.0

    # pitch-over 结束允许的弹道倾角误差，单位 degree。
    launch_pitch_over_exit_theta_error_deg: float = 3.0

    # pitch-over 结束允许的最大弹道倾角，单位 degree。
    launch_pitch_over_exit_theta_max_deg: float = 25.0

    # 是否要求进入中制导前已经处于接近几何。
    launch_pitch_over_require_closing: bool = False

    # ============================================================
    # 11. 工程化中末制导 mid_terminal_interceptor 参数
    # ============================================================

    # 完整拦截弹最大机动过载，单位 g。
    # 注意：当 interceptor_ability_profile 不是 custom 时，该值会被能力档位覆盖。
    interceptor_max_overload: float = 8.0

    # mid_terminal_interceptor 中制导比例导引系数。
    interceptor_midcourse_navigation_gain: float = 4.0

    # mid_terminal_interceptor 纵向弹道成型增益。
    interceptor_midcourse_theta_shaping_gain: float = 0.07

    # mid_terminal_interceptor 侧向弹道成型增益。
    interceptor_midcourse_psi_shaping_gain: float = 0.0

    # 完整拦截弹对红方横向机动的前馈补偿系数。
    interceptor_target_compensation_gain: float = 4.0

    # 完整拦截弹末制导比例导引系数。
    interceptor_terminal_navigation_gain: float = 6.0

    # 末制导距离切换阈值，单位 m。
    interceptor_terminal_distance_threshold: float = 10000.0

    # 末制导剩余时间切换阈值，单位 s。
    interceptor_terminal_tgo_threshold: float = 6.0

    # 完整拦截弹纵向 / 高度通道开关。
    interceptor_enable_vertical_channel: bool = True

    # 完整拦截弹侧向通道开关。
    interceptor_enable_lateral_channel: bool = True

    # ============================================================
    # 12. 论文风格 paper_mid_terminal 专用参数
    # ============================================================

    # paper_mid_terminal 专用中制导比例导引系数。
    # 说明：
    #     只作用于 guidance_mode="paper_mid_terminal"；
    #     不影响 mid_terminal_interceptor。
    paper_midcourse_navigation_gain: float = 4.0

    # paper_mid_terminal 纵向弹道成型偏置增益。
    paper_midcourse_theta_bias_gain: float = 0.07

    # paper_mid_terminal 侧向弹道成型偏置增益。
    paper_midcourse_psi_bias_gain: float = 0.09

    # paper_mid_terminal 中制导期望弹道成型角，单位 degree。
    # 论文设定常见表述为中制导阶段由大弹道倾角逐步塑形到约 20°。
    paper_midcourse_shaping_angle_deg: float = 20.0

    # paper_mid_terminal 中制导时间尺度修正系数。
    # 数值越大，中制导塑形相对更缓；数值越小，塑形相对更激进。
    paper_midcourse_time_scale: float = 1.0

    # paper_mid_terminal 纵向几何自适应弹道成型过渡带，单位 m。
    paper_altitude_transition_band: float = 5000.0

    # paper_mid_terminal 接近同高度时的中性爬升偏置比例。
    paper_neutral_climb_bias_ratio: float = 0.30

    # paper_mid_terminal 拦截弹高于红方时的向下修正偏置比例。
    paper_downward_bias_ratio: float = 0.70

    # paper_mid_terminal 侧向成型符号模式。
    # 可选：
    #     fixed_initial：
    #         每枚弹在 reset 时根据初始 dz 固定 +1 / -1 / 0；
    #     dynamic_dz：
    #         兼容旧逻辑，每一步根据当前 dz 判断；
    #     none：
    #         关闭侧向 ±20° 成型。
    paper_side_sign_mode: str = "fixed_initial"

    # paper_mid_terminal 中心线侧向符号。
    # 当初始 dz 接近 0 时，用该值指定中心线情况下的侧向成型方向。
    # 0 表示不强行指定中心线侧向偏置。
    paper_centerline_side_sign: float = 0.0

    # paper_mid_terminal 专用末制导比例导引系数。
    paper_terminal_navigation_gain: float = 6.0

    # paper_mid_terminal 末制导是否使用 cos(theta) 修正。
    paper_terminal_use_cos_theta: bool = True

    # ============================================================
    # 13. 拦截弹错过 / 回合终止判据
    # ============================================================

    # 是否使用 range-rate 判定拦截弹错过。
    # True：
    #     当距离已过最近点且正在回升时，结合 passed_distance_margin 判定错过；
    # False：
    #     使用 x 方向越界阈值 termination_x_margin 作为简化错过判据。
    use_range_rate_passed_termination: bool = True

    # 距离从历史最小值回升超过该余量后，才确认拦截弹错过，单位 m。
    passed_distance_margin: float = 50.0

    # 关闭 range-rate 判据时使用的 x 方向错过阈值，单位 m。
    termination_x_margin: float = -50.0

    # ============================================================
    # 14. 端到端奖励函数参数
    # ============================================================

    # 端到端奖励设计原则：
    #     生存 / 突防成功优先，其次才是接近任务目标和节省机动能耗。

    # 过程动作惩罚权重。
    # 数值越大，越抑制红方高频 / 大幅机动。
    reward_stage_action_weight: float = 0.03

    # 过程目标接近奖励权重。
    # 数值越大，越鼓励红方向目标方向推进。
    reward_stage_target_weight: float = 0.002

    # 终端大脱靶距离奖励线性斜率。
    # 当前形式通常为：-slope * miss_distance + bias。
    # 注意：若 miss_distance 很大，该项可能变成较大的负值，需要结合 success_bonus 检查。
    reward_terminal_large_miss_slope: float = 0.02

    # 终端大脱靶距离奖励偏置。
    reward_terminal_large_miss_bias: float = 50.6

    # 理想脱靶距离区间内的终端奖励斜率。
    reward_terminal_ideal_miss_slope: float = 1.0

    # 理想脱靶距离区间内的终端奖励偏置。
    reward_terminal_ideal_miss_bias: float = 20.0

    # 旧版终端失败惩罚。
    # 当前版本中仍保留该字段，便于兼容早期奖励设计。
    reward_terminal_failure_penalty: float = 30.0

    # 全局终端被拦截惩罚。
    # 建议在 reward.py 中显式用于 intercepted=True 的终端失败情形。
    reward_terminal_intercept_penalty: float = 250.0

    # 全局终端突防成功奖励。
    # 建议在 reward.py 中显式用于 all_passed=True 或 target_reached=True 的成功情形。
    reward_terminal_success_bonus: float = 200.0

    # 理想脱靶距离，单位 m。
    # 通常用于区分“擦边突防”和“大幅偏离任务弹道”的奖励尺度。
    reward_ideal_miss_distance: float = 50.0

    # 达到时间上限但既没有明确突防也没有被拦截时的终端惩罚。
    # 作用：防止 SAC 学到“拖到超时”而不是主动完成突防。
    reward_terminal_time_limit_penalty: float = 80.0

    # 单枚弹脱靶量 shaping 的下限。
    # 作用：避免突防成功但 miss 很大时，线性负奖励压过 success_bonus。
    reward_terminal_distance_reward_clip_min: float = -50.0

    # 单枚弹脱靶量 shaping 的上限。
    # 作用：避免脱靶量 shaping 过强，盖过 intercepted / passed 的任务成败奖励。
    reward_terminal_distance_reward_clip_max: float = 120.0

    # 多枚拦截弹脱靶量 shaping 是否取平均。
    # True：单弹和双弹奖励尺度更一致；False：沿用逐枚相加。
    reward_use_mean_terminal_distance_reward: bool = True


class PursueEscapeEnv(gym.Env):
    """
    高超声速飞行器一对二突防环境。

    Gymnasium 接口：
        reset() -> observation, info
        step()  -> observation, reward, terminated, truncated, info
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: Optional[PursueEscapeEnvConfig] = None, **kwargs: Any) -> None:
        """
        初始化环境。

        参数：
            config：
                环境配置对象；如果为 None 则使用默认配置。
            kwargs：
                允许通过关键字覆盖 config 字段。
        """
        super().__init__()

        # base_config：基础配置。
        base_config = config if config is not None else PursueEscapeEnvConfig()

        # config_dict：合并 dataclass 配置和关键字覆盖。
        config_dict = asdict(base_config)
        config_dict.update(kwargs)

        # self.config：最终环境配置。
        self.config = PursueEscapeEnvConfig(**config_dict)
        self._validate_config()
        self._apply_ability_profile()

        # state_dim：默认论文端到端 10 维状态。
        self.state_dim = self._resolve_state_dim()

        # action_dim：红方仍只有 1 维横向过载动作。
        self.action_dim = 1

        # max_steps：单回合最大步数。
        self.max_steps = int(np.ceil(self.config.t / self.config.dt))

        # action_space：红方横向过载动作空间。
        self.action_space = spaces.Box(
            low=np.array([-self.config.nzc_h_max], dtype=np.float32),
            high=np.array([self.config.nzc_h_max], dtype=np.float32),
            dtype=np.float32,
        )

        # observation_space：10 维。
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32,
        )

        # 子配置统一从 self.config 派生。
        self.navigation_config = self._build_navigation_config()
        self.reward_config = self._build_reward_config()
        self.interceptor_config = self._build_interceptor_config()
        self.interceptor_fleet = self._build_interceptor_fleet()

        # random_generator：环境内部随机数生成器。
        self.random_generator = np.random.default_rng()

        # target_position：预设打击目标三维坐标。
        self.target_position = np.array(
            [
                self.config.target_initial_x,
                self.config.target_initial_y,
                self.config.target_initial_z,
            ],
            dtype=np.float64,
        )

        # 以下变量在 reset() 中初始化。
        self.red_state = np.zeros(9, dtype=np.float64)
        self.interceptor_states: List[np.ndarray] = []
        self.current_step = 0
        self.current_time = 0.0
        self.red_inertial_overload = 0.0
        self.min_distance = np.inf
        self.initial_interceptor_distances: List[float] = []
        self.initial_target_distance = 1.0
        self.previous_target_distance = 1.0

        # 轨迹与诊断 trace。
        self._reset_traces()

    def _build_navigation_config(self) -> ProportionalNavigationConfig:
        """
        根据环境总配置生成 source_pn 制导配置。
        """
        return ProportionalNavigationConfig(
            navigation_constant=self.config.N,
            max_overload=self.config.source_pn_max_overload,
            tau=self.config.tau_i,
            compensation_gain=self.config.source_pn_compensation_gain,
            autopilot_rate_limit=self.config.interceptor_autopilot_rate_limit,
            dynamics_integration_mode=self.config.dynamics_integration_mode,
        )

    def _build_reward_config(self) -> RewardConfig:
        """
        根据环境总配置生成奖励函数配置。
        """
        return RewardConfig(
            stage_action_weight=self.config.reward_stage_action_weight,
            stage_target_weight=self.config.reward_stage_target_weight,

            terminal_large_miss_slope=self.config.reward_terminal_large_miss_slope,
            terminal_large_miss_bias=self.config.reward_terminal_large_miss_bias,
            terminal_ideal_miss_slope=self.config.reward_terminal_ideal_miss_slope,
            terminal_ideal_miss_bias=self.config.reward_terminal_ideal_miss_bias,
            terminal_failure_penalty=self.config.reward_terminal_failure_penalty,

            terminal_intercept_penalty=self.config.reward_terminal_intercept_penalty,
            terminal_success_bonus=self.config.reward_terminal_success_bonus,
            terminal_time_limit_penalty=self.config.reward_terminal_time_limit_penalty,

            ideal_miss_distance=self.config.reward_ideal_miss_distance,
            terminal_distance_reward_clip_min=self.config.reward_terminal_distance_reward_clip_min,
            terminal_distance_reward_clip_max=self.config.reward_terminal_distance_reward_clip_max,
            use_mean_terminal_distance_reward=self.config.reward_use_mean_terminal_distance_reward,
            kill_radius=self.config.kill_radius,
        )

    def _build_interceptor_config(self) -> InterceptorConfig:
        """
        根据环境总配置生成完整中末制导拦截弹配置。
        """
        return InterceptorConfig(
            max_overload=self.config.interceptor_max_overload,
            tau=self.config.tau_i,
            kill_radius=self.config.kill_radius,

            enable_launch_pitch_over=self.config.enable_launch_pitch_over,
            launch_pitch_over_guidance_modes=self.config.launch_pitch_over_guidance_modes,
            launch_pitch_over_activation_theta_deg=self.config.launch_pitch_over_activation_theta_deg,
            launch_pitch_over_fixed_theta_deg=self.config.launch_pitch_over_fixed_theta_deg,
            launch_pitch_over_blend_start_theta_deg=self.config.launch_pitch_over_blend_start_theta_deg,
            launch_pitch_over_blend_end_theta_deg=self.config.launch_pitch_over_blend_end_theta_deg,
            launch_pitch_over_los_theta_min_deg=self.config.launch_pitch_over_los_theta_min_deg,
            launch_pitch_over_los_theta_max_deg=self.config.launch_pitch_over_los_theta_max_deg,
            launch_pitch_over_theta_gain=self.config.launch_pitch_over_theta_gain,
            launch_pitch_over_vertical_overload_limit=self.config.launch_pitch_over_vertical_overload_limit,
            launch_pitch_over_lateral_overload_command=self.config.launch_pitch_over_lateral_overload_command,
            launch_pitch_over_min_duration=self.config.launch_pitch_over_min_duration,
            launch_pitch_over_max_duration=self.config.launch_pitch_over_max_duration,
            launch_pitch_over_min_altitude=self.config.launch_pitch_over_min_altitude,
            launch_pitch_over_exit_theta_error_deg=self.config.launch_pitch_over_exit_theta_error_deg,
            launch_pitch_over_exit_theta_max_deg=self.config.launch_pitch_over_exit_theta_max_deg,
            launch_pitch_over_require_closing=self.config.launch_pitch_over_require_closing,

            midcourse_navigation_gain=self.config.interceptor_midcourse_navigation_gain,
            midcourse_theta_shaping_gain=self.config.interceptor_midcourse_theta_shaping_gain,
            midcourse_psi_shaping_gain=self.config.interceptor_midcourse_psi_shaping_gain,

            paper_midcourse_navigation_gain=self.config.paper_midcourse_navigation_gain,
            paper_midcourse_theta_bias_gain=self.config.paper_midcourse_theta_bias_gain,
            paper_midcourse_psi_bias_gain=self.config.paper_midcourse_psi_bias_gain,
            paper_midcourse_shaping_angle_deg=self.config.paper_midcourse_shaping_angle_deg,
            paper_midcourse_time_scale=self.config.paper_midcourse_time_scale,
            paper_altitude_transition_band=self.config.paper_altitude_transition_band,
            paper_neutral_climb_bias_ratio=self.config.paper_neutral_climb_bias_ratio,
            paper_downward_bias_ratio=self.config.paper_downward_bias_ratio,
            paper_side_sign_mode=self.config.paper_side_sign_mode,
            paper_centerline_side_sign=self.config.paper_centerline_side_sign,

            target_compensation_gain=self.config.interceptor_target_compensation_gain,

            terminal_navigation_gain=self.config.interceptor_terminal_navigation_gain,
            paper_terminal_navigation_gain=self.config.paper_terminal_navigation_gain,
            paper_terminal_use_cos_theta=self.config.paper_terminal_use_cos_theta,

            terminal_distance_threshold=self.config.interceptor_terminal_distance_threshold,
            terminal_tgo_threshold=self.config.interceptor_terminal_tgo_threshold,

            enable_vertical_channel=self.config.interceptor_enable_vertical_channel,
            enable_lateral_channel=self.config.interceptor_enable_lateral_channel,

            autopilot_rate_limit=self.config.interceptor_autopilot_rate_limit,
            dynamics_integration_mode=self.config.dynamics_integration_mode,
        )

    def _build_interceptor_fleet(self) -> InterceptorFleet:
        """
        根据环境总配置生成拦截弹编队管理器。
        """
        return InterceptorFleet(
            guidance_mode=self.config.guidance_mode,
            navigation_config=self.navigation_config,
            interceptor_config=self.interceptor_config,
            kill_radius=self.config.kill_radius,
            passed_distance_margin=self.config.passed_distance_margin,
            use_range_rate_passed_termination=self.config.use_range_rate_passed_termination,
            termination_x_margin=self.config.termination_x_margin,
        )

    def _validate_config(self) -> None:
        """
        校验环境配置是否在支持范围内。

        返回：
            None。
        """
        if self.config.interceptor_count not in {1, 2}:
            raise ValueError(f"interceptor_count 只支持 1 或 2，当前为 {self.config.interceptor_count}")

        if self.config.observation_mode != "thesis_end_to_end_10d":
            raise ValueError(f"当前只支持 thesis_end_to_end_10d 观测，实际为 {self.config.observation_mode}")

        if self.config.guidance_mode not in SUPPORTED_GUIDANCE_MODES:
            raise ValueError(
                f"未知蓝方制导模式：{self.config.guidance_mode}，"
                f"当前支持：{sorted(SUPPORTED_GUIDANCE_MODES)}"
            )

        if self.config.dynamics_integration_mode not in {
            "semi_implicit_euler",
            "explicit_euler",
            "rk4",
        }:
            raise ValueError(
                f"未知动力学积分模式：{self.config.dynamics_integration_mode}"
            )

        if self.config.scenario_profile not in {
            "paper_200km_end_to_end",
            "manual_pair",
            "custom",
        }:
            raise ValueError(f"未知初始场景 profile：{self.config.scenario_profile}")

    def _apply_ability_profile(self) -> None:
        """
        根据能力档位更新蓝方可用过载。

        返回：
            None。
        """
        # profile：蓝方能力档位。
        profile = str(self.config.interceptor_ability_profile)

        if profile == "custom":
            return

        if profile == "weak":
            max_overload = 6.0
        elif profile == "paper":
            max_overload = 8.0
        elif profile == "strong":
            max_overload = 12.0
        else:
            raise ValueError(f"未知蓝方能力档位：{profile}")

        # source_pn 和完整拦截弹使用同一档位上限，便于公平调试。
        self.config.source_pn_max_overload = max_overload
        self.config.interceptor_max_overload = max_overload

    def _resolve_state_dim(self) -> int:
        """
        根据观测模式返回状态维度。

        返回：
            state_dim：
                当前环境观测维度。
        """
        if self.config.observation_mode == "thesis_end_to_end_10d":
            return 10

        raise ValueError(f"未知观测模式：{self.config.observation_mode}")

    def _reset_traces(self) -> None:
        """
        清空当前回合轨迹与诊断记录。

        返回：
            None。
        """
        # red_trajectory：红方位置轨迹。
        self.red_trajectory: List[np.ndarray] = []

        # interceptor_trajectories：每枚拦截弹的位置轨迹。
        self.interceptor_trajectories: List[List[np.ndarray]] = [
            [] for _ in range(int(self.config.interceptor_count))
        ]

        # red_control_trace：红方动作指令记录。
        self.red_control_trace: List[float] = []

        # red_inertial_control_trace：红方一阶实际过载记录。
        self.red_inertial_control_trace: List[float] = []

        # distance_trace：所有拦截弹中的当前最近距离记录。
        self.distance_trace: List[float] = []

        # reward_trace/time_trace/info_trace：通用回合诊断记录。
        self.reward_trace: List[float] = []
        self.time_trace: List[float] = []
        self.info_trace: List[Dict[str, Any]] = []

        # interceptor_*_traces：每枚弹双通道控制和阶段记录。
        self.interceptor_ny_command_traces: List[List[float]] = [
            [] for _ in range(int(self.config.interceptor_count))
        ]
        self.interceptor_nz_command_traces: List[List[float]] = [
            [] for _ in range(int(self.config.interceptor_count))
        ]
        self.interceptor_ny_actual_traces: List[List[float]] = [
            [] for _ in range(int(self.config.interceptor_count))
        ]
        self.interceptor_nz_actual_traces: List[List[float]] = [
            [] for _ in range(int(self.config.interceptor_count))
        ]
        self.interceptor_theta_traces: List[List[float]] = [
            [] for _ in range(int(self.config.interceptor_count))
        ]
        self.interceptor_phase_traces: List[List[str]] = [
            [] for _ in range(int(self.config.interceptor_count))
        ]

    def _sample_mach(self, fixed_value: float, min_value: Optional[float], max_value: Optional[float]) -> float:
        """
        采样 Mach 数。

        参数：
            fixed_value：
                固定 Mach 数。
            min_value/max_value：
                可选随机范围。

        返回：
            mach：
                当前回合使用的 Mach 数。
        """
        if min_value is not None and max_value is not None:
            # mach：课程训练中可在范围内随机速度。
            mach = float(self.random_generator.uniform(float(min_value), float(max_value)))
        else:
            # mach：默认使用固定速度。
            mach = float(fixed_value)

        return mach

    def _build_red_initial_state(self) -> np.ndarray:
        """
        构造红方初始状态。

        返回：
            red_state：
                红方状态向量 [x, y, z, V, theta, psi, nx, ny, nz]。
        """
        # red_mach：当前回合红方 Mach 数。
        red_mach = self._sample_mach(
            fixed_value=self.config.red_mach,
            min_value=self.config.red_mach_min,
            max_value=self.config.red_mach_max,
        )

        # red_speed：红方初始速度，单位 m/s。
        red_speed = red_mach * self.config.sound_speed

        # red_psi_delta_deg：红方初始航向随机拉偏。
        red_psi_delta_deg = float(
            self.random_generator.uniform(
                self.config.red_initial_psi_delta_min_deg,
                self.config.red_initial_psi_delta_max_deg,
            )
        )

        # red_theta/red_psi：红方初始航迹角。
        red_theta = degrees_to_radians(float(self.config.red_initial_theta_deg))
        red_psi = degrees_to_radians(float(self.config.red_initial_psi_deg + red_psi_delta_deg))

        # red_state：红方初始状态向量。
        red_state = np.zeros(9, dtype=np.float64)
        red_state[0] = float(self.config.red_initial_x)
        red_state[1] = float(self.config.red_initial_y)
        red_state[2] = float(self.config.red_initial_z)
        red_state[3] = red_speed
        red_state[4] = red_theta
        red_state[5] = red_psi
        red_state[6] = 0.0
        red_state[7] = 1.0
        red_state[8] = 0.0

        return red_state

    def _paper_profile_positions(
            self,
            x_distance: float,
            lateral_offset: float = 10000.0,
            y_offset_from_red: float = 0.0,
    ) -> List[np.ndarray]:
        """
        根据论文一对二夹击态势生成拦截弹基准位置。

        参数：
            x_distance：
                拦截弹相对红方的前向距离，单位 m。

            lateral_offset：
                双弹横向夹击偏置绝对值，单位 m。
                interceptor_count=2 时，两枚弹分别位于 -offset 和 +offset。

            y_offset_from_red：
                拦截弹相对红方的高度偏置，单位 m。
                interceptor_y = red_initial_y + y_offset_from_red。

        返回：
            positions：
                每枚弹的三维位置列表。
        """
        if self.config.interceptor_count == 1:
            z_offsets = [0.0]
        else:
            z_offsets = [-float(lateral_offset), float(lateral_offset)]

        positions = [
            np.array(
                [
                    float(self.config.red_initial_x + x_distance),
                    float(self.config.red_initial_y + y_offset_from_red),
                    float(self.config.red_initial_z + z_offset),
                ],
                dtype=np.float64,
            )
            for z_offset in z_offsets
        ]

        return positions

    def _manual_pair_profile_positions(self) -> List[np.ndarray]:
        """
        根据 manual_pair profile 生成手动指定的拦截弹初始位置。

        返回：
            positions：
                每枚弹的三维位置列表。
        """
        positions = [
            np.array(
                [
                    float(self.config.manual_interceptor_1_x),
                    float(self.config.manual_interceptor_1_y),
                    float(self.config.manual_interceptor_1_z),
                ],
                dtype=np.float64,
            )
        ]

        if self.config.interceptor_count >= 2:
            positions.append(
                np.array(
                    [
                        float(self.config.manual_interceptor_2_x),
                        float(self.config.manual_interceptor_2_y),
                        float(self.config.manual_interceptor_2_z),
                    ],
                    dtype=np.float64,
                )
            )

        return positions

    def _custom_profile_positions(self) -> List[np.ndarray]:
        """
        根据自定义随机范围生成拦截弹初始位置。

        返回：
            positions：
                每枚弹的三维位置列表。
        """
        # positions：自定义 profile 下每枚弹独立采样。
        positions: List[np.ndarray] = []

        for _ in range(int(self.config.interceptor_count)):
            # x/y/z：自定义位置范围内采样。
            x = float(
                self.random_generator.uniform(
                    self.config.interceptor_initial_x_min,
                    self.config.interceptor_initial_x_max,
                )
            )
            y = float(
                self.random_generator.uniform(
                    self.config.interceptor_initial_y_min,
                    self.config.interceptor_initial_y_max,
                )
            )
            z = float(
                self.random_generator.uniform(
                    self.config.interceptor_initial_z_min,
                    self.config.interceptor_initial_z_max,
                )
            )
            positions.append(np.array([x, y, z], dtype=np.float64))

        return positions

    def _base_interceptor_positions(self) -> List[np.ndarray]:
        """
        根据 scenario_profile 生成拦截弹基准位置。

        返回：
            positions：
                每枚弹的三维基准位置。
        """
        if self.config.scenario_profile == "paper_200km_end_to_end":
            return self._paper_profile_positions(
                x_distance=float(self.config.paper_interceptor_x_distance),
                lateral_offset=float(self.config.paper_interceptor_lateral_offset),
                y_offset_from_red=float(self.config.paper_interceptor_y_offset_from_red),
            )

        if self.config.scenario_profile == "manual_pair":
            return self._manual_pair_profile_positions()

        if self.config.scenario_profile == "custom":
            return self._custom_profile_positions()

        raise ValueError(f"未知初始场景 profile：{self.config.scenario_profile}")

    def _sample_interceptor_heading_deg(self, interceptor_index: int) -> float:
        """
        采样指定拦截弹的初始航向角。

        参数：
            interceptor_index：
                拦截弹编号，从 1 开始。

        返回：
            heading_deg：
                当前拦截弹初始航向角，单位 degree。
        """
        if self.config.scenario_profile == "manual_pair":
            if interceptor_index == 1:
                base_heading_deg = float(self.config.manual_interceptor_1_heading_deg)
            else:
                base_heading_deg = float(self.config.manual_interceptor_2_heading_deg)

        elif self.config.scenario_profile == "custom":
            base_heading_deg = float(
                self.random_generator.uniform(
                    self.config.interceptor_initial_heading_min_deg,
                    self.config.interceptor_initial_heading_max_deg,
                )
            )

        else:
            # paper/debug profile：默认迎头飞向红方。
            base_heading_deg = 180.0

        if self.config.initial_randomization_enabled:
            heading_delta_deg = float(
                self.random_generator.uniform(
                    -self.config.interceptor_heading_randomization_deg,
                    self.config.interceptor_heading_randomization_deg,
                )
            )
        else:
            heading_delta_deg = 0.0

        return float(base_heading_deg + heading_delta_deg)

    def _sample_interceptor_theta_deg(self, interceptor_index: int) -> float:
        """
        采样指定拦截弹的初始弹道倾角。

        参数：
            interceptor_index：
                拦截弹编号，从 1 开始。

        返回：
            theta_deg：
                当前拦截弹初始弹道倾角，单位 degree。
        """
        if self.config.scenario_profile == "manual_pair":
            if interceptor_index == 1:
                base_theta_deg = float(self.config.manual_interceptor_1_theta_deg)
            else:
                base_theta_deg = float(self.config.manual_interceptor_2_theta_deg)
        elif self.config.scenario_profile == "paper_200km_end_to_end":
            base_theta_deg = float(self.config.paper_interceptor_theta_deg)
        else:
            base_theta_deg = float(self.config.interceptor_initial_theta_deg)

        if self.config.initial_randomization_enabled:
            theta_delta_deg = float(
                self.random_generator.uniform(
                    -self.config.interceptor_theta_randomization_deg,
                    self.config.interceptor_theta_randomization_deg,
                )
            )
        else:
            theta_delta_deg = 0.0

        return float(base_theta_deg + theta_delta_deg)

    def _build_interceptor_initial_states(self) -> List[np.ndarray]:
        """
        构造所有拦截弹初始状态。

        返回：
            states：
                拦截弹状态向量列表。
        """
        # interceptor_mach：当前回合拦截弹 Mach 数。
        interceptor_mach = self._sample_mach(
            fixed_value=self.config.interceptor_mach,
            min_value=self.config.interceptor_mach_min,
            max_value=self.config.interceptor_mach_max,
        )

        # interceptor_speed：拦截弹初始速度，单位 m/s。
        interceptor_speed = interceptor_mach * self.config.sound_speed

        # base_positions：根据场景 profile 得到的基准位置。
        base_positions = self._base_interceptor_positions()

        # states：最终拦截弹初始状态列表。
        states: List[np.ndarray] = []

        for interceptor_index, base_position in enumerate(base_positions, start=1):
            # position：复制基准位置，避免原数组被随机扰动污染。
            position = base_position.copy()

            if self.config.initial_randomization_enabled and self.config.scenario_profile != "custom":
                # position_delta：
                #     对 paper/debug/manual_pair 的基准位置加入随机扰动；
                #     custom profile 本身已经在范围内随机采样，所以不再叠加二次扰动。
                position_delta = self.random_generator.uniform(
                    -self.config.interceptor_position_randomization_m,
                    self.config.interceptor_position_randomization_m,
                    size=3,
                )

                if not bool(self.config.randomize_interceptor_y):
                    position_delta[1] = 0.0

                position = position + position_delta

            # theta/psi：支持每枚拦截弹独立初始角配置。
            theta = degrees_to_radians(
                self._sample_interceptor_theta_deg(interceptor_index)
            )
            psi = degrees_to_radians(
                self._sample_interceptor_heading_deg(interceptor_index)
            )

            # state：单枚拦截弹初始状态。
            state = np.zeros(9, dtype=np.float64)
            state[0:3] = position
            state[3] = interceptor_speed
            state[4] = theta
            state[5] = psi
            state[6] = 0.0
            state[7] = 1.0
            state[8] = 0.0
            states.append(state)

        return states

    def _target_distance(self) -> float:
        """
        计算红方到预设打击目标的距离。

        返回：
            distance：
                红方到目标的三维距离，单位 m。
        """
        # distance：红方当前位置与目标位置的欧氏距离。
        distance = float(np.linalg.norm(self.target_position - self.red_state[:3]))

        return distance

    def _observation_for_interceptor(self, interceptor_state: np.ndarray, slot_index: int) -> List[float]:
        """
        计算单枚拦截弹对应的 4 个论文观测分量。

        参数：
            interceptor_state：
                拦截弹当前状态。
            slot_index：
                拦截弹槽位索引，从 0 开始，用于读取初始距离归一化尺度。

        返回：
            features：
                [r, dr, q, dq] 四个归一化特征。
        """
        # relative_info：红方相对当前拦截弹的几何信息。
        relative_info = compute_relative_geometry(
            red_state=self.red_state,
            interceptor_state=interceptor_state,
        )

        # initial_distance：当前槽位初始距离，用作 r 归一化尺度。
        initial_distance = max(float(self.initial_interceptor_distances[slot_index]), EPS)

        # distance：当前相对距离。
        distance = float(relative_info["distance"])

        # range_rate：signed 距离变化率；小于 0 表示正在接近。
        range_rate = float(relative_info["range_rate"])

        # relative_speed_scale：论文中使用 V_H + V_I 对距离变化率无量纲化。
        relative_speed_scale = max(float(self.red_state[3] + interceptor_state[3]), EPS)

        # los_angle：水平 X-Z 平面视线角。
        los_angle = wrap_angle(float(relative_info["los_angle"]))

        # los_rate：水平 X-Z 平面视线角速度。
        los_rate = float(relative_info["los_rate_standard"])

        # features：按论文第 5 章状态定义归一化。
        features = [
            float(np.clip(distance / initial_distance, 0.0, 2.0)),
            float(np.clip(range_rate / relative_speed_scale, -1.0, 1.0)),
            float(np.clip(los_angle / (2.0 * np.pi), -1.0, 1.0)),
            float(np.clip(los_rate, -1.0, 1.0)),
        ]

        return features

    def _get_observation(self) -> np.ndarray:
        """
        获取当前 10 维论文式端到端观测。

        返回：
            observation：
                shape=(10,) 的 float32 观测向量。
        """
        # features：按 [I1四维, I2四维, rHT, nHz] 组装。
        features: List[float] = []

        for slot_index in range(2):
            if slot_index < len(self.interceptor_states):
                # 当前槽位存在真实拦截弹。
                features.extend(self._observation_for_interceptor(self.interceptor_states[slot_index], slot_index))
            else:
                # 单弹回归模式下第二个槽位补 0，保持状态维度为 10。
                features.extend([0.0, 0.0, 0.0, 0.0])

        # target_distance_norm：红方到预设打击目标距离归一化。
        target_distance_norm = float(np.clip(self._target_distance() / max(self.initial_target_distance, EPS), 0.0, 2.0))

        # red_overload_norm：红方实际横向过载归一化。
        red_overload_norm = float(np.clip(self.red_inertial_overload / max(self.config.nzc_h_max, EPS), -1.0, 1.0))

        features.append(target_distance_norm)
        features.append(red_overload_norm)

        # observation：最终 10 维观测。
        observation = np.asarray(features, dtype=np.float32)

        return observation

    def _record_step(self, reward: float, info: Dict[str, Any]) -> None:
        """
        记录当前步轨迹、控制量和诊断信息。

        参数：
            reward：
                当前步奖励。
            info：
                当前步诊断字典。

        返回：
            None。
        """
        # red_trajectory：记录红方三维位置。
        self.red_trajectory.append(self.red_state[:3].copy())

        for index, state in enumerate(self.interceptor_states, start=1):
            # trajectory：记录第 index 枚拦截弹三维位置。
            self.interceptor_trajectories[index - 1].append(state[:3].copy())

            # prefix：该弹字段前缀。
            prefix = f"interceptor_{index}"

            # interceptor_*_traces：记录该弹双通道控制和阶段。
            self.interceptor_ny_command_traces[index - 1].append(float(info.get(f"{prefix}_ny_command", 1.0)))
            self.interceptor_nz_command_traces[index - 1].append(float(info.get(f"{prefix}_nz_command", 0.0)))
            self.interceptor_ny_actual_traces[index - 1].append(float(info.get(f"{prefix}_ny_actual", 1.0)))
            self.interceptor_nz_actual_traces[index - 1].append(float(info.get(f"{prefix}_nz_actual", 0.0)))
            self.interceptor_theta_traces[index - 1].append(float(info.get(f"{prefix}_theta", 0.0)))
            self.interceptor_phase_traces[index - 1].append(str(info.get(f"{prefix}_phase", "unknown")))

        # 通用 trace：红方控制、距离、奖励、时间和 info。
        self.red_control_trace.append(float(info.get("red_command_overload", 0.0)))
        self.red_inertial_control_trace.append(float(info.get("red_inertial_overload", 0.0)))
        self.distance_trace.append(float(info.get("distance", np.nan)))
        self.reward_trace.append(float(reward))
        self.time_trace.append(float(self.current_time))
        self.info_trace.append(dict(info))

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        重置环境。

        参数：
            seed：
                随机种子。
            options：
                预留选项，本阶段不使用。

        返回：
            observation：
                初始观测。
            info：
                初始诊断信息。
        """
        if seed is not None:
            # random_generator：重置环境内部随机数。
            self.random_generator = np.random.default_rng(seed)
            super().reset(seed=seed)
        else:
            super().reset(seed=None)

        # red_state：红方初始状态。
        self.red_state = self._build_red_initial_state()

        # interceptor_states：所有拦截弹初始状态。
        self.interceptor_states = self._build_interceptor_initial_states()

        # interceptor_fleet：重置蓝方编队。
        self.interceptor_fleet.reset(
            red_state=self.red_state,
            initial_states=self.interceptor_states,
        )
        self.interceptor_states = self.interceptor_fleet.states

        # current_step/current_time：回合计时器。
        self.current_step = 0
        self.current_time = 0.0

        # red_inertial_overload：红方实际横向过载初始为 0。
        self.red_inertial_overload = 0.0

        # initial_interceptor_distances：每枚弹初始距离，用于观测归一化。
        self.initial_interceptor_distances = [
            float(np.linalg.norm(state[:3] - self.red_state[:3]))
            for state in self.interceptor_states
        ]

        # min_distance：全局连续最小距离。
        self.min_distance = min(self.initial_interceptor_distances) if self.initial_interceptor_distances else np.inf

        # initial_target_distance/previous_target_distance：目标距离归一化和奖励过程项基准。
        self.initial_target_distance = max(self._target_distance(), EPS)
        self.previous_target_distance = self.initial_target_distance

        # trace：清空并等待 step 记录。
        self._reset_traces()

        # observation：初始 10 维观测。
        observation = self._get_observation()

        # info：初始诊断字段。
        info: Dict[str, Any] = {
            "time": self.current_time,
            "step": self.current_step,
            "min_distance": float(self.min_distance),
            "target_distance": float(self.initial_target_distance),
            "interceptor_count": int(len(self.interceptor_states)),
            "guidance_mode": self.config.guidance_mode,
            "observation_mode": self.config.observation_mode,
            "scenario_profile": self.config.scenario_profile,
        }

        for index, state in enumerate(self.interceptor_states, start=1):
            # prefix：该弹初始字段前缀。
            prefix = f"interceptor_{index}"
            info[f"{prefix}_initial_x"] = float(state[0])
            info[f"{prefix}_initial_y"] = float(state[1])
            info[f"{prefix}_initial_z"] = float(state[2])
            info[f"{prefix}_initial_psi"] = float(state[5])
            info[f"{prefix}_min_distance"] = float(self.initial_interceptor_distances[index - 1])

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        执行一步环境交互。

        参数：
            action：
                红方动作，形状为 [1]，表示横向过载指令。

        返回：
            observation：
                下一状态观测。
            reward：
                当前步奖励。
            terminated：
                是否自然终止。
            truncated：
                是否达到最大步数截断。
            info：
                当前步诊断信息。
        """
        # action_array：转换并拉平后的动作数组。
        action_array = np.asarray(action, dtype=np.float64).reshape(-1)

        if action_array.shape[0] != self.action_dim:
            raise ValueError(f"动作维度错误，应为 {self.action_dim}，实际为 {action_array.shape[0]}")

        # previous_red_position：更新前红方位置，用于连续最近点判据。
        previous_red_position = self.red_state[:3].copy()

        # previous_target_distance：更新前红方到目标距离，用于过程奖励。
        previous_target_distance = self._target_distance()

        # red_command_overload：红方横向过载指令限幅。
        red_command_overload = float(np.clip(action_array[0], -self.config.nzc_h_max, self.config.nzc_h_max))

        # red_autopilot_output/red_autopilot_info：红方一阶自动驾驶仪输出和工程诊断。
        red_autopilot_output, red_autopilot_info = FirstOrderAutopilot.compute_response_with_info(
            command=red_command_overload,
            previous_output=self.red_inertial_overload,
            dt=self.config.dt,
            tau=self.config.tau_h,
            rate_limit=self.config.red_autopilot_rate_limit,
            output_min=-self.config.nzc_h_max,
            output_max=self.config.nzc_h_max,
        )
        self.red_inertial_overload = float(red_autopilot_output)
        self.red_inertial_overload = float(
            np.clip(self.red_inertial_overload, -self.config.nzc_h_max, self.config.nzc_h_max)
        )

        # red_state：红方质点状态更新。
        self.red_state = update_point_mass_state(
            state=self.red_state,
            nx=0.0,
            ny=1.0,
            nz=self.red_inertial_overload,
            dt=self.config.dt,
            integration_mode=self.config.dynamics_integration_mode,
        )
        self.red_state[6] = 0.0
        self.red_state[7] = 1.0
        self.red_state[8] = self.red_inertial_overload

        # current_step/current_time：推进仿真时间。
        self.current_step += 1
        self.current_time = self.current_step * self.config.dt

        # fleet_info：推进所有拦截弹并收集诊断字段。
        fleet_info = self.interceptor_fleet.step(
            red_state=self.red_state,
            previous_red_position=previous_red_position,
            red_lateral_overload=self.red_inertial_overload,
            dt=self.config.dt,
            current_step=self.current_step,
        )
        self.interceptor_states = self.interceptor_fleet.states

        # min_distance：全局连续最小距离。
        self.min_distance = min(float(self.min_distance), float(fleet_info["min_distance"]))

        # intercepted/all_passed：多弹终止判据。
        intercepted = bool(fleet_info.get("intercepted", False))
        all_passed = bool(fleet_info.get("all_interceptors_passed", False))

        # terminated：任一弹命中或全部弹错过均自然终止。
        terminated = bool(intercepted or all_passed)

        # truncated：达到最大步数时截断。
        truncated = bool((self.current_step >= self.max_steps) and not terminated)

        if intercepted:
            termination_reason = "intercepted"
        elif all_passed:
            termination_reason = "passed"
        elif truncated:
            termination_reason = "time_limit"
        else:
            termination_reason = "running"

        # current_target_distance：更新后红方到目标距离。
        current_target_distance = self._target_distance()

        # interceptor_min_distances：每枚弹的连续最小脱靶量。
        interceptor_min_distances = [
            float(fleet_info[f"interceptor_{index}_min_distance"])
            for index in range(1, len(self.interceptor_states) + 1)
        ]

        # reward：端到端奖励。
        # 说明：
        #     奖励函数需要显式接收 intercepted / all_passed / truncated / termination_reason，
        #     否则无法区分“被拦截失败”“全部错过成功”和“拖到时间上限”。
        reward, reward_info = calculate_end_to_end_reward(
            red_lateral_overload=self.red_inertial_overload,
            previous_target_distance=previous_target_distance,
            current_target_distance=current_target_distance,
            interceptor_min_distances=interceptor_min_distances,
            terminated=terminated,
            config=self.reward_config,
            truncated=truncated,
            intercepted=intercepted,
            all_passed=all_passed,
            termination_reason=termination_reason,
        )

        # info：当前步综合诊断字段。
        info: Dict[str, Any] = {
            **fleet_info,
            "time": float(self.current_time),
            "step": int(self.current_step),
            "red_command_overload": float(red_command_overload),
            "red_inertial_overload": float(self.red_inertial_overload),
            "red_autopilot_rate_saturated": bool(red_autopilot_info["rate_saturated"]),
            "red_autopilot_output_saturated": bool(red_autopilot_info["output_saturated"]),
            "target_distance": float(current_target_distance),
            "previous_target_distance": float(previous_target_distance),
            "min_distance": float(self.min_distance),
            "terminated": terminated,
            "truncated": truncated,
            "intercepted": intercepted,
            "passed": all_passed,
            "success": bool(termination_reason == "passed"),
            "termination_reason": termination_reason,
            "guidance_mode": self.config.guidance_mode,
            "observation_mode": self.config.observation_mode,
            "scenario_profile": self.config.scenario_profile,
        }
        info.update(reward_info)

        # previous_target_distance：保存给下一步使用。
        self.previous_target_distance = current_target_distance

        # trace：记录轨迹和诊断。
        self._record_step(reward=reward, info=info)

        # observation：下一状态 10 维观测。
        observation = self._get_observation()

        return observation, float(reward), terminated, truncated, info

    def get_episode_metrics(self) -> Dict[str, float]:
        """
        获取当前回合基础统计指标。

        返回：
            metrics：
                当前回合统计字典。
        """
        # total_reward：当前回合累计奖励。
        total_reward = float(np.sum(self.reward_trace)) if self.reward_trace else 0.0

        # final_distance：最后一步全局最近当前距离。
        final_distance = float(self.distance_trace[-1]) if self.distance_trace else float(self.min_distance)

        # last_info：最后一步诊断信息，用于区分真正突防成功和训练器提前截断。
        last_info = self.info_trace[-1] if self.info_trace else {}

        # success：只有环境自然判定全部拦截弹错过时才算突防成功。
        success = float(bool(last_info.get("success", False)))

        # metrics：基础回合指标。
        metrics = {
            "total_reward": total_reward,
            "min_distance": float(self.min_distance),
            "final_distance": final_distance,
            "success": success,
            "interceptor_count": float(len(self.interceptor_states)),
            "episode_steps": float(self.current_step),
            "episode_time": float(self.current_time),
        }

        return metrics

    def render(self) -> None:
        """
        打印当前环境状态。

        返回：
            None。
        """
        print(
            f"step={self.current_step}, "
            f"time={self.current_time:.2f}, "
            f"red_x={self.red_state[0]:.2f}, "
            f"min_distance={self.min_distance:.2f}"
        )
