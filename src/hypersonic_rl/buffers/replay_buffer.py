"""
replay_buffer.py

作用：
    实现 SAC 使用的普通经验回放池 ReplayBuffer。

经验回放池保存的数据格式：
    state      当前状态
    action     当前动作
    reward     当前奖励
    next_state 下一状态
    done       当前回合是否结束

工程说明：
    经验回放池不依赖具体环境，只关心状态维度和动作维度。
    后续 SAC、TD3、DDPG 等连续动作强化学习算法都可以复用该模块。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch


@dataclass
class ReplayBufferConfig:
    """
    ReplayBufferConfig

    作用：
        管理经验回放池的基础配置。

    属性：
        state_dim：
            状态向量维度。例如当前环境中，状态由红方和蓝方前 6 个状态拼接，
            因此默认训练环境 state_dim = 10。

        action_dim：
            动作向量维度。例如当前环境中，红方横向过载控制是一维动作，
            因此 action_dim = 1。

        capacity：
            经验回放池最大容量。超过容量后，旧数据会被循环覆盖。

        device：
            采样后张量放置的设备，例如 "cpu" 或 "cuda:0"。
    """

    state_dim: int
    action_dim: int
    capacity: int
    device: str = "cpu"


class ReplayBuffer:
    """
    SAC 普通经验回放池。

    设计目标：
        1. 使用 numpy 数组进行高效存储；
        2. 采样时转换为 torch.Tensor；
        3. 支持循环覆盖，避免内存无限增长；
        4. 对外返回标准 batch 字典，方便 SAC Agent 使用。
    """

    def __init__(self, config: ReplayBufferConfig) -> None:
        """
        初始化经验回放池。

        参数：
            config：
                ReplayBufferConfig 配置对象。
        """
        # state_dim：状态维度。
        self.state_dim = int(config.state_dim)

        # action_dim：动作维度。
        self.action_dim = int(config.action_dim)

        # capacity：经验池最大容量。
        self.capacity = int(config.capacity)

        # device：采样后张量所在设备。
        self.device = torch.device(config.device)

        if self.state_dim <= 0:
            raise ValueError(f"state_dim 必须大于 0，当前为：{self.state_dim}")

        if self.action_dim <= 0:
            raise ValueError(f"action_dim 必须大于 0，当前为：{self.action_dim}")

        if self.capacity <= 0:
            raise ValueError(f"capacity 必须大于 0，当前为：{self.capacity}")

        # states：存储当前状态，形状为 [capacity, state_dim]。
        self.states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)

        # actions：存储当前动作，形状为 [capacity, action_dim]。
        self.actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)

        # rewards：存储奖励，形状为 [capacity, 1]。
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)

        # next_states：存储下一状态，形状为 [capacity, state_dim]。
        self.next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)

        # dones：存储终止标志，1 表示回合结束，0 表示未结束。
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)

        # pointer：当前写入位置。超过 capacity 后会从 0 重新开始覆盖。
        self.pointer = 0

        # size：当前已经存储的样本数量，最大不超过 capacity。
        self.size = 0

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        添加一条环境交互经验。

        参数：
            state：
                当前状态数组，形状应为 [state_dim]。

            action：
                当前动作数组，形状应为 [action_dim]。

            reward：
                当前动作获得的奖励。

            next_state：
                执行动作后的下一状态数组，形状应为 [state_dim]。

            done：
                当前回合是否结束。
        """
        # state_array：转换并拉平后的当前状态。
        state_array = np.asarray(state, dtype=np.float32).reshape(-1)

        # action_array：转换并拉平后的当前动作。
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)

        # next_state_array：转换并拉平后的下一状态。
        next_state_array = np.asarray(next_state, dtype=np.float32).reshape(-1)

        if state_array.shape[0] != self.state_dim:
            raise ValueError(f"state 维度错误，应为 {self.state_dim}，实际为 {state_array.shape[0]}")

        if action_array.shape[0] != self.action_dim:
            raise ValueError(f"action 维度错误，应为 {self.action_dim}，实际为 {action_array.shape[0]}")

        if next_state_array.shape[0] != self.state_dim:
            raise ValueError(f"next_state 维度错误，应为 {self.state_dim}，实际为 {next_state_array.shape[0]}")

        # index：本次写入的数组索引。
        index = self.pointer

        self.states[index] = state_array
        self.actions[index] = action_array
        self.rewards[index] = float(reward)
        self.next_states[index] = next_state_array
        self.dones[index] = float(done)

        # pointer：写入指针循环后移。
        self.pointer = (self.pointer + 1) % self.capacity

        # size：经验池有效样本数量。
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        随机采样一个 batch。

        参数：
            batch_size：
                采样数量。

        返回：
            batch：
                包含 state、action、reward、next_state、done 的字典。
        """
        if self.size == 0:
            raise RuntimeError("经验回放池为空，无法采样。")

        if batch_size <= 0:
            raise ValueError(f"batch_size 必须大于 0，当前为：{batch_size}")

        if batch_size > self.size:
            raise ValueError(f"batch_size={batch_size} 大于当前经验池样本数 size={self.size}")

        # indices：随机采样得到的样本索引。
        indices = np.random.randint(low=0, high=self.size, size=batch_size)

        # batch：转换为 torch.Tensor 后的样本字典。
        batch = {
            "state": torch.as_tensor(self.states[indices], dtype=torch.float32, device=self.device),
            "action": torch.as_tensor(self.actions[indices], dtype=torch.float32, device=self.device),
            "reward": torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=self.device),
            "next_state": torch.as_tensor(self.next_states[indices], dtype=torch.float32, device=self.device),
            "done": torch.as_tensor(self.dones[indices], dtype=torch.float32, device=self.device),
        }

        return batch

    def clear(self) -> None:
        """
        清空经验回放池。

        说明：
            该函数不会重新分配内存，只会重置 pointer 和 size。
        """
        self.pointer = 0
        self.size = 0

    def __len__(self) -> int:
        """
        返回当前经验池有效样本数量。
        """
        return self.size

    @property
    def is_full(self) -> bool:
        """
        判断经验回放池是否已满。
        """
        return self.size >= self.capacity
