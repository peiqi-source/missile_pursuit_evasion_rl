"""环境包装器，预留第 5 章 LSTM-SAC 状态序列接口。"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


class StateSequenceWrapper:
    """将单步观测包装为固定长度状态序列。

    第 5 章 LSTM-SAC 需要输入历史状态序列而不是单个状态。当前包装器提供基础
    reset/step 接口，后续可与 SequenceReplayBuffer 和 LSTMActor 联动。
    """

    def __init__(self, env: Any, sequence_length: int = 8) -> None:
        self.env = env
        self.sequence_length = int(sequence_length)
        self.state_queue: deque[np.ndarray] = deque(maxlen=self.sequence_length)
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self, *args: Any, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        """重置环境并用初始观测填满状态序列。"""
        observation, info = self.env.reset(*args, **kwargs)
        self.state_queue.clear()
        for _ in range(self.sequence_length):
            self.state_queue.append(np.asarray(observation, dtype=np.float32))
        return self._sequence(), info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """执行一步并返回状态序列观测。"""
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.state_queue.append(np.asarray(observation, dtype=np.float32))
        return self._sequence(), reward, terminated, truncated, info

    def _sequence(self) -> np.ndarray:
        """返回形状为 [sequence_length, obs_dim] 的状态序列。"""
        return np.stack(list(self.state_queue), axis=0).astype(np.float32)

    def __getattr__(self, name: str) -> Any:
        """透传未显式实现的环境属性。"""
        return getattr(self.env, name)


class ObservationNormalizeWrapper:
    """观测归一化包装器骨架。

    当前阶段默认透传观测；后续若训练不稳定，可接入 RunningMeanStd 对状态做在线归一化。
    """

    def __init__(self, env: Any, epsilon: float = 1e-8) -> None:
        self.env = env
        self.epsilon = float(epsilon)
        self.action_space = env.action_space
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

