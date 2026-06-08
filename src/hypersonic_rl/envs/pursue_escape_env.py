"""Gymnasium 风格的高超声速追逃环境。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hypersonic_rl.envs.dynamics import (
    apply_first_order_lag,
    compute_relative_distance,
    update_point_mass_state,
)
from hypersonic_rl.envs.guidance import proportional_navigation_guidance
from hypersonic_rl.envs.reward import compute_end_to_end_sac_reward

try:  # pragma: no cover - 真实环境安装 gymnasium 时使用官方空间定义。
    from gymnasium import spaces
except Exception:  # pragma: no cover - 测试环境缺少 gymnasium 时使用轻量降级版本。

    class _Box:
        """最小 Box 空间实现，用于避免测试环境缺少 gymnasium 时失败。"""

        def __init__(self, low: Any, high: Any, shape: tuple[int, ...], dtype: Any) -> None:
            self.low = np.full(shape, low, dtype=dtype)
            self.high = np.full(shape, high, dtype=dtype)
            self.shape = shape
            self.dtype = dtype

        def sample(self) -> np.ndarray:
            return np.random.uniform(self.low, self.high).astype(self.dtype)

    class spaces:  # type: ignore[no-redef]
        Box = _Box


@dataclass
class EnvConfig:
    """环境配置默认值。"""

    nzc_h_max: float = 3.0
    nzc_i_max: float = 6.0
    dt: float = 0.01
    max_time: float = 20.0
    tau_h: float = 0.5
    tau_i: float = 0.5
    navigation_constant: float = 4.0
    kill_radius: float = 7.0
    gravity: float = 9.81


class PursueEscapeEnv:
    """红方突防飞行器与蓝方拦截弹的端到端 SAC 环境。

    当前环境保留原始代码逻辑：Actor 直接输出红方横向过载，蓝方使用比例导引。
    后续若切换到论文第 4 章范式，可在制导模块中让 Actor 输出微分对策制导律参数。
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        # config：环境超参数字典，可由 YAML 读取后传入。
        self.config_dict = config or {}
        cfg = EnvConfig(**{k: v for k, v in self.config_dict.items() if hasattr(EnvConfig, k)})
        self.nzc_h_max = float(cfg.nzc_h_max)
        self.nzc_i_max = float(cfg.nzc_i_max)
        # dt：仿真积分步长。
        self.dt = float(cfg.dt)
        self.max_time = float(cfg.max_time)
        # tau_h：红方自动驾驶仪一阶惯性时间常数。
        self.tau_h = float(cfg.tau_h)
        # tau_i：蓝方拦截弹制导控制一阶惯性时间常数。
        self.tau_i = float(cfg.tau_i)
        self.navigation_constant = float(cfg.navigation_constant)
        self.kill_radius = float(cfg.kill_radius)
        self.gravity = float(cfg.gravity)
        self.max_steps = max(1, int(round(self.max_time / self.dt)))

        # observation_space：观测空间，包含红蓝双方各 6 个运动状态。
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )
        # action_space：动作空间，1 维红方横向机动过载。
        self.action_space = spaces.Box(
            low=-self.nzc_h_max,
            high=self.nzc_h_max,
            shape=(1,),
            dtype=np.float32,
        )
        self.rng = np.random.default_rng()
        self.reset()

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """重置环境并返回初始观测。

        参数:
            seed: 随机种子，用于复现蓝方初始位置和航向。
            options: 预留 Gymnasium reset 选项。

        返回:
            observation: 初始观测向量。
            info: 环境辅助信息。
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        _ = options

        red_cfg = self.config_dict.get("red_initial_state", {})
        blue_cfg = self.config_dict.get("blue_initial_random", {})

        # red_state：红方飞行器状态向量，包含位置、速度、弹道倾角、弹道偏角和过载。
        self.red_state = np.array(
            [
                float(red_cfg.get("x", 0.0)),
                float(red_cfg.get("y", 27000.0)),
                float(red_cfg.get("z", 0.0)),
                float(red_cfg.get("speed", 1870.0)),
                float(red_cfg.get("theta", 0.0)),
                float(red_cfg.get("psi", 0.0)),
                0.0,
                1.0,
                0.0,
            ],
            dtype=np.float64,
        )
        blue_x = self.rng.uniform(
            float(blue_cfg.get("x_min", 28000.0)),
            float(blue_cfg.get("x_max", 32000.0)),
        )
        blue_psi_deg = self.rng.uniform(
            float(blue_cfg.get("psi_min_deg", 175.0)),
            float(blue_cfg.get("psi_max_deg", 180.0)),
        )
        # blue_state：蓝方拦截弹状态向量。
        self.blue_state = np.array(
            [
                blue_x,
                float(blue_cfg.get("y", 27000.0)),
                float(blue_cfg.get("z", 0.0)),
                float(blue_cfg.get("speed", 1190.0)),
                0.0,
                np.deg2rad(blue_psi_deg),
                0.0,
                1.0,
                0.0,
            ],
            dtype=np.float64,
        )

        # action：智能体输出的红方横向机动控制量。
        self.last_action = np.zeros(1, dtype=np.float64)
        self.inertial_action = np.zeros(1, dtype=np.float64)
        self.last_nzc_i = np.zeros(1, dtype=np.float64)
        self.inertial_nzc_i = np.zeros(1, dtype=np.float64)
        self.nzc_i = np.zeros(1, dtype=np.float64)
        self.current_step = 0
        self.current_time = 0.0
        self._reset_trace()
        observation = self._get_observation()
        info = {"time": self.current_time, "distance": compute_relative_distance(self.red_state, self.blue_state)}
        return observation, info

    def step(self, action: np.ndarray | float) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """执行一步环境交互。

        参数:
            action: 智能体输出的红方横向机动控制量。

        返回:
            observation, reward, terminated, truncated, info。
        """
        raw_action = np.asarray(action, dtype=np.float64).reshape(1)
        clipped_action = np.clip(raw_action, -self.nzc_h_max, self.nzc_h_max)

        red_control, inertial_action = self._compute_red_control(clipped_action)
        self.red_state = update_point_mass_state(
            self.red_state, red_control, dt=self.dt, gravity=self.gravity
        )
        blue_control, blue_command, inertial_blue, guidance_info = proportional_navigation_guidance(
            red_state=self.red_state,
            blue_state=self.blue_state,
            red_lateral_acceleration=inertial_action,
            previous_blue_output=self.last_nzc_i,
            dt=self.dt,
            tau_i=self.tau_i,
            nzc_i_max=self.nzc_i_max,
            navigation_constant=self.navigation_constant,
            gravity=self.gravity,
        )
        self.nzc_i = blue_command.copy()
        self.inertial_nzc_i = inertial_blue.copy()
        self.last_nzc_i = inertial_blue.copy()
        self.blue_state = update_point_mass_state(
            self.blue_state, blue_control, dt=self.dt, gravity=self.gravity
        )

        distance = compute_relative_distance(self.red_state, self.blue_state)
        self.distance_trace.append(distance)
        reward, reward_info = compute_end_to_end_sac_reward(
            action=clipped_action,
            distance_trace=self.distance_trace,
            red_state=self.red_state,
            blue_state=self.blue_state,
            dqz=guidance_info["dqz"],
            kill_radius=self.kill_radius,
        )
        self.current_step += 1
        self.current_time = self.current_step * self.dt
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_steps
        observation = self._get_observation()
        self._record_trace(clipped_action, reward, reward_info)
        info = {
            "time": self.current_time,
            "distance": distance,
            "min_distance": min(self.distance_trace),
            "guidance": guidance_info,
            "reward_info": reward_info,
        }
        return observation, float(reward), bool(terminated), bool(truncated), info

    def render(self) -> None:
        """打印当前 episode 的最小距离，用于轻量调试。"""
        if self.distance_trace:
            min_distance = min(self.distance_trace)
            print(f"min_distance: {min_distance:.3f}")

    def close(self) -> None:
        """关闭环境占用资源。

        当前环境没有外部仿真进程或图形窗口，因此该函数为空实现。
        """

    def get_episode_trace(self) -> dict[str, np.ndarray]:
        """返回统一格式的 episode 轨迹数据。"""
        return {
            "time": np.asarray(self.time_trace, dtype=np.float64),
            "red_states": np.asarray(self.save_red_states, dtype=np.float64),
            "blue_states": np.asarray(self.save_blue_states, dtype=np.float64),
            "red_trajectory": np.asarray(self.red_trajectory, dtype=np.float64),
            "blue_trajectory": np.asarray(self.blue_trajectory, dtype=np.float64),
            "actions": np.asarray(self.control_trace, dtype=np.float64).reshape(-1),
            "inertial_actions": np.asarray(self.inertial_control_trace, dtype=np.float64).reshape(-1),
            "blue_commands": np.asarray(self.nzc_i_trace, dtype=np.float64).reshape(-1),
            "inertial_blue_commands": np.asarray(self.inertial_nzc_i_trace, dtype=np.float64).reshape(-1),
            "distances": np.asarray(self.distance_trace, dtype=np.float64),
            "rewards": np.asarray(self.reward_trace, dtype=np.float64),
            "reward_terms": np.asarray(self.reward_terms, dtype=object),
            "dt": np.asarray([self.dt], dtype=np.float64),
        }

    def _compute_red_control(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """计算红方一阶惯性过载输出。"""
        self.inertial_action = apply_first_order_lag(
            self.last_action, action, dt=self.dt, tau=self.tau_h
        ).reshape(1)
        self.last_action = self.inertial_action.copy()
        red_control = np.array([0.0, 1.0, float(self.inertial_action[0])], dtype=np.float64)
        return red_control, self.inertial_action.copy()

    def _get_observation(self) -> np.ndarray:
        """拼接红蓝双方前 6 个状态作为观测。"""
        observation = np.concatenate((self.red_state[:6], self.blue_state[:6]))
        return observation.astype(np.float32)

    def _is_terminated(self) -> bool:
        """判断 episode 是否自然终止。"""
        relative_x = float(self.blue_state[0] - self.red_state[0])
        return relative_x < -50.0

    def _reset_trace(self) -> None:
        """清空 episode 轨迹缓存。"""
        self.time_trace: list[float] = []
        self.red_trajectory: list[tuple[float, float]] = []
        self.blue_trajectory: list[tuple[float, float]] = []
        self.control_trace: list[float] = []
        self.inertial_control_trace: list[float] = []
        self.nzc_i_trace: list[float] = []
        self.inertial_nzc_i_trace: list[float] = []
        self.distance_trace: list[float] = []
        self.reward_trace: list[float] = []
        self.reward_terms: list[dict[str, float]] = []
        self.save_red_states: list[np.ndarray] = []
        self.save_blue_states: list[np.ndarray] = []

    def _record_trace(
        self, action: np.ndarray, reward: float, reward_info: dict[str, float]
    ) -> None:
        """记录当前仿真步的轨迹、控制量、距离和奖励明细。"""
        self.time_trace.append(self.current_time)
        self.red_trajectory.append((float(self.red_state[0]), float(self.red_state[2])))
        self.blue_trajectory.append((float(self.blue_state[0]), float(self.blue_state[2])))
        self.control_trace.append(float(action[0]))
        self.inertial_control_trace.append(float(self.inertial_action[0]))
        self.nzc_i_trace.append(float(self.nzc_i[0]))
        self.inertial_nzc_i_trace.append(float(self.inertial_nzc_i[0]))
        self.reward_trace.append(float(reward))
        self.reward_terms.append(dict(reward_info))
        self.save_red_states.append(self.red_state.copy())
        self.save_blue_states.append(self.blue_state.copy())

