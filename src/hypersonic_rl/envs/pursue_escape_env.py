"""
pursue_escape_env.py

作用：
    实现高超声速飞行器追逃突防强化学习环境。

环境角色：
    红方：
        高超声速飞行器，由 SAC 智能体控制。

    蓝方：
        拦截弹，使用比例导引追击红方。

环境接口：
    reset():
        重置环境，返回初始观测。

    step(action):
        执行动作，返回下一状态、奖励、终止标志和信息字典。

观测定义：
    observation = concat(red_state[:6], blue_state[:6])
    因此当前观测维度为 12。

动作定义：
    action = [nzc_h]
    其中 nzc_h 是红方侧向过载控制指令。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from hypersonic_rl.envs.dynamics import degrees_to_radians, update_point_mass_state
from hypersonic_rl.envs.guidance import (
    ProportionalNavigationConfig,
    apply_first_order_lag,
    compute_blue_proportional_navigation,
)
from hypersonic_rl.envs.reward import RewardConfig, calculate_pursuit_escape_reward


@dataclass
class PursueEscapeEnvConfig:
    """
    PursueEscapeEnvConfig

    作用：
        管理追逃突防环境参数。

    属性：
        nzc_h_max：
            红方最大侧向过载。

        nzc_i_max：
            蓝方最大侧向过载。

        dt：
            仿真步长，单位 s。

        t:
            单回合最大仿真时间，单位 s。

        tau_h：
            红方一阶惯性时间常数。

        tau_i：
            蓝方一阶惯性时间常数。

        N：
            蓝方比例导引系数。

        kill_radius：
            杀伤半径，单位 m。

        sound_speed：
            声速，单位 m/s。
            当前用于把 Mach 数转换为 m/s。

        red_mach：
            红方初始 Mach 数。

        blue_mach：
            蓝方初始 Mach 数。

        initial_altitude：
            初始高度，单位 m。

        red_initial_x：
            红方初始 x 坐标。

        blue_initial_x_min / blue_initial_x_max：
            蓝方初始 x 坐标随机范围。

        blue_initial_heading_min_deg / blue_initial_heading_max_deg：
            蓝方初始航向角随机范围，单位 degree。

        termination_x_margin：
            终止判据。
            当 blue_x - red_x < termination_x_margin 时，认为蓝方已经越过红方。
    """

    nzc_h_max: float = 3.0
    nzc_i_max: float = 6.0
    dt: float = 0.01
    t: float = 20.0
    tau_h: float = 0.5
    tau_i: float = 0.5
    N: float = 4.0
    kill_radius: float = 7.0
    sound_speed: float = 340.0
    red_mach: float = 5.5
    blue_mach: float = 3.5
    initial_altitude: float = 27000.0
    red_initial_x: float = 0.0
    blue_initial_x_min: float = 28000.0
    blue_initial_x_max: float = 32000.0
    blue_initial_heading_min_deg: float = 175.0
    blue_initial_heading_max_deg: float = 180.0
    termination_x_margin: float = -50.0


class PursueEscapeEnv(gym.Env):
    """
    高超声速飞行器追逃突防环境。

    该环境遵循 Gymnasium 风格返回：
        reset() -> observation, info
        step()  -> observation, reward, terminated, truncated, info

    如果后续使用旧版 Gym 训练器，只需要在 trainer 中兼容返回值即可。
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: Optional[PursueEscapeEnvConfig] = None, **kwargs: Any) -> None:
        """
        初始化环境。

        参数：
            config：
                PursueEscapeEnvConfig 配置对象。
                如果为 None，则使用默认参数。

            kwargs：
                允许通过关键字覆盖 config 中的字段。
                例如 PursueEscapeEnv(dt=0.02, nzc_h_max=4.0)。
        """
        super().__init__()

        # base_config：基础配置。
        base_config = config if config is not None else PursueEscapeEnvConfig()

        # config_dict：配置字典，用于处理 kwargs 覆盖。
        config_dict = base_config.__dict__.copy()
        config_dict.update(kwargs)

        # self.config：最终环境配置。
        self.config = PursueEscapeEnvConfig(**config_dict)

        # state_dim：观测状态维度，红方前 6 维 + 蓝方前 6 维。
        self.state_dim = 12

        # action_dim：动作维度，当前红方只有一个侧向过载动作。
        self.action_dim = 1

        # max_steps：单回合最大步数。
        self.max_steps = int(np.ceil(self.config.t / self.config.dt))

        # action_space：动作空间。
        self.action_space = spaces.Box(
            low=np.array([-self.config.nzc_h_max], dtype=np.float32),
            high=np.array([self.config.nzc_h_max], dtype=np.float32),
            dtype=np.float32,
        )

        # observation_space：观测空间。
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32,
        )

        # navigation_config：蓝方比例导引配置。
        self.navigation_config = ProportionalNavigationConfig(
            navigation_constant=self.config.N,
            max_overload=self.config.nzc_i_max,
            tau=self.config.tau_i,
            compensation_gain=0.5,
        )

        # reward_config：奖励函数配置。
        self.reward_config = RewardConfig(kill_radius=self.config.kill_radius)

        # random_generator：环境内部随机数生成器。
        self.random_generator = np.random.default_rng()

        # 以下变量在 reset() 中初始化。
        self.red_state = np.zeros(9, dtype=np.float64)
        self.blue_state = np.zeros(9, dtype=np.float64)
        self.current_step = 0
        self.current_time = 0.0
        self.red_inertial_overload = 0.0
        self.blue_inertial_overload = 0.0
        self.min_distance = np.inf

        # 初始化轨迹记录列表。
        self._reset_traces()

    def _reset_traces(self) -> None:
        """
        清空当前回合轨迹与日志记录。
        """
        # red_trajectory：红方位置轨迹。
        self.red_trajectory = []

        # blue_trajectory：蓝方位置轨迹。
        self.blue_trajectory = []

        # red_control_trace：红方原始动作指令记录。
        self.red_control_trace = []

        # red_inertial_control_trace：红方一阶惯性后实际过载记录。
        self.red_inertial_control_trace = []

        # blue_control_trace：蓝方比例导引原始指令记录。
        self.blue_control_trace = []

        # blue_inertial_control_trace：蓝方一阶惯性后实际过载记录。
        self.blue_inertial_control_trace = []

        # distance_trace：红蓝距离记录。
        self.distance_trace = []

        # reward_trace：奖励记录。
        self.reward_trace = []

        # time_trace：时间记录。
        self.time_trace = []

        # info_trace：每一步诊断信息记录。
        self.info_trace = []

    def _build_red_initial_state(self) -> np.ndarray:
        """
        构造红方初始状态。

        返回：
            red_state：
                红方初始状态向量，形状为 [9]。
        """
        # red_speed：红方初始速度，单位 m/s。
        red_speed = self.config.red_mach * self.config.sound_speed

        # red_state：红方状态向量。
        red_state = np.zeros(9, dtype=np.float64)
        red_state[0] = self.config.red_initial_x
        red_state[1] = self.config.initial_altitude
        red_state[2] = 0.0
        red_state[3] = red_speed
        red_state[4] = 0.0
        red_state[5] = 0.0

        return red_state

    def _build_blue_initial_state(self) -> np.ndarray:
        """
        构造蓝方初始状态。

        返回：
            blue_state：
                蓝方初始状态向量，形状为 [9]。
        """
        # blue_speed：蓝方初始速度，单位 m/s。
        blue_speed = self.config.blue_mach * self.config.sound_speed

        # blue_x：蓝方初始 x 坐标，在配置范围内随机。
        blue_x = self.random_generator.uniform(
            self.config.blue_initial_x_min,
            self.config.blue_initial_x_max,
        )

        # blue_heading_deg：蓝方初始航向角，单位 degree。
        blue_heading_deg = self.random_generator.uniform(
            self.config.blue_initial_heading_min_deg,
            self.config.blue_initial_heading_max_deg,
        )

        # blue_state：蓝方状态向量。
        blue_state = np.zeros(9, dtype=np.float64)
        blue_state[0] = blue_x
        blue_state[1] = self.config.initial_altitude
        blue_state[2] = 0.0
        blue_state[3] = blue_speed
        blue_state[4] = 0.0
        blue_state[5] = degrees_to_radians(blue_heading_deg)

        return blue_state

    def _get_observation(self) -> np.ndarray:
        """
        获取当前观测。

        返回：
            observation：
                拼接后的 12 维观测向量。
        """
        # observation：红方前 6 维和蓝方前 6 维拼接。
        observation = np.concatenate([self.red_state[:6], self.blue_state[:6]], axis=0)

        return observation.astype(np.float32)

    def _record_step(self, reward: float, info: Dict[str, float]) -> None:
        """
        记录当前步轨迹、控制量和诊断信息。

        参数：
            reward：
                当前步奖励。

            info：
                当前步信息字典。
        """
        self.red_trajectory.append(self.red_state[:3].copy())
        self.blue_trajectory.append(self.blue_state[:3].copy())
        self.red_control_trace.append(float(info.get("red_command_overload", 0.0)))
        self.red_inertial_control_trace.append(float(info.get("red_inertial_overload", 0.0)))
        self.blue_control_trace.append(float(info.get("blue_command_overload", 0.0)))
        self.blue_inertial_control_trace.append(float(info.get("blue_inertial_overload", 0.0)))
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
                预留选项字典，当前暂不使用。

        返回：
            observation：
                初始观测。

            info：
                初始信息字典。
        """
        if seed is not None:
            self.random_generator = np.random.default_rng(seed)

            try:
                super().reset(seed=seed)
            except TypeError:
                pass

        # red_state：红方初始状态。
        self.red_state = self._build_red_initial_state()

        # blue_state：蓝方初始状态。
        self.blue_state = self._build_blue_initial_state()

        # current_step：当前回合步数。
        self.current_step = 0

        # current_time：当前仿真时间。
        self.current_time = 0.0

        # red_inertial_overload：红方一阶惯性后实际过载。
        self.red_inertial_overload = 0.0

        # blue_inertial_overload：蓝方一阶惯性后实际过载。
        self.blue_inertial_overload = 0.0

        # min_distance：当前回合最小红蓝距离。
        self.min_distance = float(np.linalg.norm(self.blue_state[:3] - self.red_state[:3]))

        self._reset_traces()

        # observation：初始观测。
        observation = self._get_observation()

        # info：初始信息。
        info = {
            "time": self.current_time,
            "step": self.current_step,
            "min_distance": self.min_distance,
        }

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        执行一步环境交互。

        参数：
            action：
                红方动作，形状为 [1]。
                表示红方侧向过载指令。

        返回：
            observation：
                下一状态观测。

            reward：
                当前步奖励。

            terminated：
                是否自然终止。

            truncated：
                是否由于达到最大步数而截断。

            info：
                诊断信息字典。
        """
        # action_array：转换并拉平后的动作数组。
        action_array = np.asarray(action, dtype=np.float64).reshape(-1)

        if action_array.shape[0] != self.action_dim:
            raise ValueError(f"动作维度错误，应为 {self.action_dim}，实际为 {action_array.shape[0]}")

        # red_command_overload：红方原始侧向过载指令。
        red_command_overload = float(np.clip(action_array[0], -self.config.nzc_h_max, self.config.nzc_h_max))

        # red_inertial_overload：红方经过一阶惯性后的实际侧向过载。
        self.red_inertial_overload = apply_first_order_lag(
            command=red_command_overload,
            previous_output=self.red_inertial_overload,
            dt=self.config.dt,
            tau=self.config.tau_h,
        )

        self.red_inertial_overload = float(
            np.clip(self.red_inertial_overload, -self.config.nzc_h_max, self.config.nzc_h_max)
        )

        # 红方状态更新。
        # nx=0 表示不主动改变速度，ny=1 表示近似维持高度，nz 为侧向机动过载。
        self.red_state = update_point_mass_state(
            state=self.red_state,
            nx=0.0,
            ny=1.0,
            nz=self.red_inertial_overload,
            dt=self.config.dt,
        )

        # 蓝方比例导引计算。
        blue_command_overload, self.blue_inertial_overload, guidance_info = compute_blue_proportional_navigation(
            red_state=self.red_state,
            blue_state=self.blue_state,
            red_lateral_overload=self.red_inertial_overload,
            previous_blue_overload=self.blue_inertial_overload,
            dt=self.config.dt,
            config=self.navigation_config,
        )

        # 蓝方状态更新。
        self.blue_state = update_point_mass_state(
            state=self.blue_state,
            nx=0.0,
            ny=1.0,
            nz=self.blue_inertial_overload,
            dt=self.config.dt,
        )

        # current_step/current_time：推进仿真时间。
        self.current_step += 1
        self.current_time = self.current_step * self.config.dt

        # current_distance：当前红蓝距离。
        current_distance = float(np.linalg.norm(self.blue_state[:3] - self.red_state[:3]))

        # min_distance：更新当前回合最小距离。
        self.min_distance = min(self.min_distance, current_distance)

        # terminated：自然终止条件。
        # 当蓝方 x 坐标已经落到红方后方一定距离，认为本次追逃交会结束。
        terminated = bool((self.blue_state[0] - self.red_state[0]) < self.config.termination_x_margin)

        # truncated：最大步数截断。
        truncated = bool(self.current_step >= self.max_steps)

        # reward：当前步奖励。
        reward, reward_info = calculate_pursuit_escape_reward(
            red_state=self.red_state,
            blue_state=self.blue_state,
            red_action=self.red_inertial_overload,
            min_distance=self.min_distance,
            terminated=terminated,
            config=self.reward_config,
        )

        # info：当前步综合信息。
        info: Dict[str, Any] = {
            "time": float(self.current_time),
            "step": int(self.current_step),
            "red_command_overload": float(red_command_overload),
            "red_inertial_overload": float(self.red_inertial_overload),
            "blue_command_overload": float(blue_command_overload),
            "blue_inertial_overload": float(self.blue_inertial_overload),
            "distance": float(current_distance),
            "min_distance": float(self.min_distance),
            "terminated": terminated,
            "truncated": truncated,
        }

        info.update(guidance_info)
        info.update(reward_info)

        self._record_step(reward=reward, info=info)

        # observation：下一状态观测。
        observation = self._get_observation()

        return observation, float(reward), terminated, truncated, info

    def get_episode_metrics(self) -> Dict[str, float]:
        """
        获取当前回合统计指标。

        返回：
            metrics：
                当前回合指标字典。
        """
        # total_reward：当前回合累计奖励。
        total_reward = float(np.sum(self.reward_trace)) if self.reward_trace else 0.0

        # final_distance：最后一步距离。
        final_distance = float(self.distance_trace[-1]) if self.distance_trace else float(self.min_distance)

        # success：是否突防成功。
        success = float(self.min_distance > self.config.kill_radius)

        metrics = {
            "total_reward": total_reward,
            "min_distance": float(self.min_distance),
            "final_distance": final_distance,
            "success": success,
            "episode_steps": float(self.current_step),
            "episode_time": float(self.current_time),
        }

        return metrics

    def render(self) -> None:
        """
        打印当前环境状态。
        """
        print(
            f"step={self.current_step}, "
            f"time={self.current_time:.2f}, "
            f"red_x={self.red_state[0]:.2f}, "
            f"blue_x={self.blue_state[0]:.2f}, "
            f"distance={self.min_distance:.2f}"
        )