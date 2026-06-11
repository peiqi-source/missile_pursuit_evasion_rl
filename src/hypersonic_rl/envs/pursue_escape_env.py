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

from dataclasses import dataclass
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
        管理一对二突防环境参数。

    关键约定：
        1. 红方动作维度固定为 1；
        2. 默认观测维度为论文第 5 章端到端 10 维状态；
        3. 拦截弹数量支持 1 或 2，默认 2；
        4. 默认场景为论文 200 km 一对二训练态势。
    """

    # 红方最大横向过载，单位 g。
    nzc_h_max: float = 2.0

    # source_pn 模式下拦截弹最大机动过载，单位 g。
    source_pn_max_overload: float = 8.0

    # 仿真步长和最大仿真时长。
    dt: float = 0.01
    t: float = 80.0

    # 红方和拦截弹自动驾驶仪一阶时间常数。
    tau_h: float = 0.5
    tau_i: float = 0.5

    # 质点运动学积分模式，默认保持当前工程半隐式欧拉行为。
    dynamics_integration_mode: str = "semi_implicit_euler"

    # 红方自动驾驶仪可选输出变化率限制，单位 g/s；None 表示不启用。
    red_autopilot_rate_limit: Optional[float] = None

    # 蓝方自动驾驶仪可选输出变化率限制，单位 g/s；None 表示不启用。
    interceptor_autopilot_rate_limit: Optional[float] = None

    # source_pn 比例导引系数。
    N: float = 4.0

    # 杀伤半径，论文默认 5m。
    kill_radius: float = 5.0

    # Mach 到 m/s 的工程换算声速。
    sound_speed: float = 340.0

    # 红方和拦截弹默认速度。
    red_mach: float = 6.0
    interceptor_mach: float = 4.0

    # 课程训练可选速度随机化范围。
    red_mach_min: Optional[float] = None
    red_mach_max: Optional[float] = None
    interceptor_mach_min: Optional[float] = None
    interceptor_mach_max: Optional[float] = None

    # 蓝方制导模式。
    guidance_mode: str = "mid_terminal_interceptor"

    # 默认观测模式。
    observation_mode: str = "thesis_end_to_end_10d"

    # 初始态势 profile。
    scenario_profile: str = "paper_200km_end_to_end"

    # 拦截弹数量，支持 1 或 2。
    interceptor_count: int = 2

    # 蓝方能力档位：weak / paper / strong / custom。
    interceptor_ability_profile: str = "paper"

    # 是否启用论文第 5 章训练随机化。
    initial_randomization_enabled: bool = True

    # 拦截弹初始位置随机扰动范围，单位 m。
    interceptor_position_randomization_m: float = 3000.0

    # 拦截弹初始航向角随机扰动范围，单位 degree。
    interceptor_heading_randomization_deg: float = 3.0

    # 红方初始位置，单位 m。
    red_initial_x: float = 0.0
    red_initial_y: float = 25000.0
    red_initial_z: float = 0.0

    # 红方初始航迹倾角和航向角，单位 degree。
    red_initial_theta_deg: float = 0.0
    red_initial_psi_deg: float = 0.0

    # 红方初始航向角拉偏，单位 degree。
    red_initial_psi_delta_min_deg: float = 0.0
    red_initial_psi_delta_max_deg: float = 0.0

    # 自定义 profile 下拦截弹初始位置随机范围。
    interceptor_initial_x_min: float = 200000.0
    interceptor_initial_x_max: float = 200000.0
    interceptor_initial_y_min: float = 25000.0
    interceptor_initial_y_max: float = 25000.0
    interceptor_initial_z_min: float = 0.0
    interceptor_initial_z_max: float = 0.0

    # 自定义 profile 下拦截弹初始航迹倾角和航向角范围。
    interceptor_initial_theta_deg: float = 0.0
    interceptor_initial_heading_min_deg: float = 180.0
    interceptor_initial_heading_max_deg: float = 180.0

    # 预设打击目标位置 (220km, 25km, 0)。
    target_initial_x: float = 220000.0
    target_initial_y: float = 25000.0
    target_initial_z: float = 0.0

    # source_pn 对红方机动的前馈补偿系数。
    source_pn_compensation_gain: float = 0.5

    # 完整拦截弹最大机动过载。
    interceptor_max_overload: float = 8.0

    # 完整拦截弹中制导比例导引和弹道成型参数。
    interceptor_midcourse_navigation_gain: float = 4.0
    interceptor_midcourse_theta_shaping_gain: float = 0.07
    interceptor_midcourse_psi_shaping_gain: float = 0.0

    # 完整拦截弹对红方横向机动的前馈补偿系数。
    interceptor_target_compensation_gain: float = 4.0

    # 完整拦截弹末制导参数。
    interceptor_terminal_navigation_gain: float = 6.0
    interceptor_terminal_distance_threshold: float = 10000.0
    interceptor_terminal_tgo_threshold: float = 6.0

    # 完整拦截弹双通道开关。
    interceptor_enable_vertical_channel: bool = True
    interceptor_enable_lateral_channel: bool = True

    # 是否使用 range-rate 判定拦截弹错过。
    use_range_rate_passed_termination: bool = True

    # 距离从历史最小值回升超过该余量后，才确认错过。
    passed_distance_margin: float = 50.0

    # 关闭 range-rate 判据时使用的 x 方向错过阈值。
    termination_x_margin: float = -50.0

    # 端到端奖励函数参数。
    reward_stage_action_weight: float = 0.03
    reward_stage_target_weight: float = 0.002
    reward_terminal_large_miss_slope: float = 0.02
    reward_terminal_large_miss_bias: float = 50.6
    reward_terminal_ideal_miss_slope: float = 1.0
    reward_terminal_ideal_miss_bias: float = 20.0
    reward_terminal_failure_penalty: float = 30.0
    reward_ideal_miss_distance: float = 50.0


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
        config_dict = base_config.__dict__.copy()
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

        # navigation_config：source_pn 模式配置。
        self.navigation_config = ProportionalNavigationConfig(
            navigation_constant=self.config.N,
            max_overload=self.config.source_pn_max_overload,
            tau=self.config.tau_i,
            compensation_gain=self.config.source_pn_compensation_gain,
            autopilot_rate_limit=self.config.interceptor_autopilot_rate_limit,
            dynamics_integration_mode=self.config.dynamics_integration_mode,
        )

        # reward_config：端到端奖励配置。
        self.reward_config = RewardConfig(
            stage_action_weight=self.config.reward_stage_action_weight,
            stage_target_weight=self.config.reward_stage_target_weight,
            terminal_large_miss_slope=self.config.reward_terminal_large_miss_slope,
            terminal_large_miss_bias=self.config.reward_terminal_large_miss_bias,
            terminal_ideal_miss_slope=self.config.reward_terminal_ideal_miss_slope,
            terminal_ideal_miss_bias=self.config.reward_terminal_ideal_miss_bias,
            terminal_failure_penalty=self.config.reward_terminal_failure_penalty,
            ideal_miss_distance=self.config.reward_ideal_miss_distance,
            kill_radius=self.config.kill_radius,
        )

        # interceptor_config：完整中末制导拦截弹配置。
        self.interceptor_config = InterceptorConfig(
            max_overload=self.config.interceptor_max_overload,
            tau=self.config.tau_i,
            kill_radius=self.config.kill_radius,
            midcourse_navigation_gain=self.config.interceptor_midcourse_navigation_gain,
            midcourse_theta_shaping_gain=self.config.interceptor_midcourse_theta_shaping_gain,
            midcourse_psi_shaping_gain=self.config.interceptor_midcourse_psi_shaping_gain,
            target_compensation_gain=self.config.interceptor_target_compensation_gain,
            terminal_navigation_gain=self.config.interceptor_terminal_navigation_gain,
            terminal_distance_threshold=self.config.interceptor_terminal_distance_threshold,
            terminal_tgo_threshold=self.config.interceptor_terminal_tgo_threshold,
            enable_vertical_channel=self.config.interceptor_enable_vertical_channel,
            enable_lateral_channel=self.config.interceptor_enable_lateral_channel,
            autopilot_rate_limit=self.config.interceptor_autopilot_rate_limit,
            dynamics_integration_mode=self.config.dynamics_integration_mode,
        )

        # interceptor_fleet：蓝方拦截弹编队。
        self.interceptor_fleet = InterceptorFleet(
            guidance_mode=self.config.guidance_mode,
            navigation_config=self.navigation_config,
            interceptor_config=self.interceptor_config,
            kill_radius=self.config.kill_radius,
            passed_distance_margin=self.config.passed_distance_margin,
            use_range_rate_passed_termination=self.config.use_range_rate_passed_termination,
            termination_x_margin=self.config.termination_x_margin,
        )

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
            "debug_50km_head_on",
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

    def _paper_profile_positions(self, x_distance: float, lateral_offset: float = 10000.0) -> List[np.ndarray]:
        """
        根据论文一对二夹击态势生成拦截弹基准位置。

        参数：
            x_distance：
                拦截弹与红方的前向距离，单位 m。
            lateral_offset：
                双弹横向夹击偏置，单位 m。

        返回：
            positions：
                每枚弹的三维位置列表。
        """
        if self.config.interceptor_count == 1:
            # 单弹回归 profile：放在中心线，便于和历史单弹 debug 对照。
            z_offsets = [0.0]
        else:
            # 双弹 profile：两枚弹位于红方两侧，偏置由具体 profile 决定。
            z_offsets = [-float(lateral_offset), float(lateral_offset)]

        # positions：按当前工程坐标约定转换后的拦截弹初始位置。
        positions = [
            np.array(
                [
                    float(self.config.red_initial_x + x_distance),
                    float(self.config.red_initial_y),
                    float(self.config.red_initial_z + z_offset),
                ],
                dtype=np.float64,
            )
            for z_offset in z_offsets
        ]

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
            return self._paper_profile_positions(x_distance=200000.0, lateral_offset=10000.0)

        if self.config.scenario_profile == "debug_50km_head_on":
            return self._paper_profile_positions(x_distance=50000.0, lateral_offset=2000.0)

        if self.config.scenario_profile == "custom":
            return self._custom_profile_positions()

        raise ValueError(f"未知初始场景 profile：{self.config.scenario_profile}")

    def _sample_interceptor_heading_deg(self) -> float:
        """
        采样拦截弹初始航向角。

        返回：
            heading_deg：
                当前拦截弹初始航向角，单位 degree。
        """
        if self.config.scenario_profile == "custom":
            # base_heading_deg：自定义 profile 下使用配置范围采样。
            base_heading_deg = float(
                self.random_generator.uniform(
                    self.config.interceptor_initial_heading_min_deg,
                    self.config.interceptor_initial_heading_max_deg,
                )
            )
        else:
            # 论文 profile：拦截弹迎头飞向红方，航向角为 180 度。
            base_heading_deg = 180.0

        if self.config.initial_randomization_enabled:
            # heading_delta_deg：论文第 5 章中拦截弹初始角度 ±3 度随机化。
            heading_delta_deg = float(
                self.random_generator.uniform(
                    -self.config.interceptor_heading_randomization_deg,
                    self.config.interceptor_heading_randomization_deg,
                )
            )
        else:
            heading_delta_deg = 0.0

        return float(base_heading_deg + heading_delta_deg)

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

        for base_position in base_positions:
            # position：复制基准位置，避免原数组被随机扰动污染。
            position = base_position.copy()

            if self.config.initial_randomization_enabled and self.config.scenario_profile != "custom":
                # position_delta：论文第 5 章发射位置 ±3km 随机生成。
                position_delta = self.random_generator.uniform(
                    -self.config.interceptor_position_randomization_m,
                    self.config.interceptor_position_randomization_m,
                    size=3,
                )
                # 高度扰动会引入额外纵向难度，本轮默认只扰动 x-z 平面。
                position_delta[1] = 0.0
                position = position + position_delta

            # theta/psi：拦截弹初始航迹倾角和航向角。
            theta = degrees_to_radians(float(self.config.interceptor_initial_theta_deg))
            psi = degrees_to_radians(self._sample_interceptor_heading_deg())

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
        reward, reward_info = calculate_end_to_end_reward(
            red_lateral_overload=self.red_inertial_overload,
            previous_target_distance=previous_target_distance,
            current_target_distance=current_target_distance,
            interceptor_min_distances=interceptor_min_distances,
            terminated=terminated,
            config=self.reward_config,
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
            "success": bool((not intercepted) and terminated),
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
