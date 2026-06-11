"""环境包装器，提供第 5 章 LSTM-SAC 状态序列接口。"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from gymnasium import spaces


class StateSequenceWrapper:
    """将单步观测包装为固定长度状态序列。

    第 5 章 LSTM-SAC 需要输入历史状态序列而不是单个状态。
    reset 时用初始观测填满窗口，step 时滚动追加最新观测。
    """

    def __init__(self, env: Any, sequence_length: int = 8) -> None:
        """
        初始化状态序列包装器。

        参数：
            env：
                被包装的原始环境对象。
            sequence_length：
                LSTM 输入所需的历史状态序列长度。

        返回：
            None。
        """
        # env：保存被包装环境，后续 reset/step 都转发到该对象。
        self.env = env

        # sequence_length：固定历史状态窗口长度。
        self.sequence_length = int(sequence_length)

        # state_queue：保存最近 sequence_length 个观测。
        self.state_queue: deque[np.ndarray] = deque(maxlen=self.sequence_length)

        # action_space：对外暴露原环境动作空间。
        self.action_space = env.action_space

        # base_observation_space：原始单步观测空间。
        base_observation_space = env.observation_space

        if not isinstance(base_observation_space, spaces.Box):
            raise TypeError("StateSequenceWrapper 当前只支持 Box 类型观测空间。")

        # observation_space：LSTM-SAC 对外观测空间，形状为 [sequence_length, obs_dim]。
        self.observation_space = spaces.Box(
            low=np.repeat(base_observation_space.low[np.newaxis, ...], self.sequence_length, axis=0),
            high=np.repeat(base_observation_space.high[np.newaxis, ...], self.sequence_length, axis=0),
            dtype=np.float32,
        )

    def reset(self, *args: Any, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        """
        重置环境并用初始观测填满状态序列。

        返回：
            sequence：
                形状为 [sequence_length, obs_dim] 的初始状态序列。
            info：
                原始环境 reset 返回的诊断信息。
        """
        # reset_result：兼容 Gymnasium tuple 返回和旧 Gym 单观测返回。
        reset_result = self.env.reset(*args, **kwargs)

        if isinstance(reset_result, tuple):
            observation, info = reset_result
        else:
            observation = reset_result
            info = {}

        self.state_queue.clear()

        for _ in range(self.sequence_length):
            # 初始窗口：用同一个初始观测重复填充，保证 episode 第一步也有完整序列。
            self.state_queue.append(np.asarray(observation, dtype=np.float32))

        return self._sequence(), info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        执行一步并返回状态序列观测。

        参数：
            action：
                传给原始环境的红方动作。

        返回：
            sequence：
                滚动更新后的状态序列。
            reward/terminated/truncated/info：
                原始环境 step 结果。
        """
        # step_result：兼容 Gymnasium 五返回值和旧 Gym 四返回值。
        step_result = self.env.step(action)

        if len(step_result) == 5:
            observation, reward, terminated, truncated, info = step_result
        else:
            observation, reward, done, info = step_result
            terminated = bool(done)
            truncated = False

        # state_queue：追加最新单步观测，最旧观测由 deque 自动弹出。
        self.state_queue.append(np.asarray(observation, dtype=np.float32))

        return self._sequence(), reward, terminated, truncated, info

    def _sequence(self) -> np.ndarray:
        """
        返回固定长度状态序列。

        返回：
            sequence：
                float32 数组，形状为 [sequence_length, obs_dim]。
        """
        if len(self.state_queue) != self.sequence_length:
            raise RuntimeError(
                f"状态序列长度错误，应为 {self.sequence_length}，实际为 {len(self.state_queue)}"
            )

        # sequence：将 deque 中的单步观测堆叠为 LSTM 输入。
        sequence = np.stack(list(self.state_queue), axis=0).astype(np.float32)

        return sequence

    def __getattr__(self, name: str) -> Any:
        """透传未显式实现的环境属性。"""
        return getattr(self.env, name)


class ObservationNormalizeWrapper:
    """观测归一化包装器骨架。

    当前阶段默认透传观测；后续若训练不稳定，可接入 RunningMeanStd 对状态做在线归一化。
    """

    def __init__(self, env: Any, epsilon: float = 1e-8) -> None:
        """
        初始化观测归一化包装器。

        参数：
            env：
                被包装的原始环境对象。
            epsilon：
                归一化时用于避免除零的小常数。

        返回：
            None。
        """
        # env：保存被包装环境。
        self.env = env

        # epsilon：预留给后续在线归一化使用的数值稳定常数。
        self.epsilon = float(epsilon)

        # action_space：保持动作空间与原环境一致。
        self.action_space = env.action_space

        # observation_space：保持观测空间与原环境一致。
        self.observation_space = env.observation_space

    def reset(self, *args: Any, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        """重置环境并返回归一化观测。"""
        observation, info = self.env.reset(*args, **kwargs)
        return self.normalize(observation), info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """执行一步并返回归一化观测。"""
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self.normalize(observation), reward, terminated, truncated, info

    def normalize(self, observation: np.ndarray) -> np.ndarray:
        """当前阶段透传观测，保留后续扩展入口。"""
        return np.asarray(observation, dtype=np.float32)

    def __getattr__(self, name: str) -> Any:
        """透传未显式实现的环境属性。"""
        return getattr(self.env, name)
