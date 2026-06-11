"""
sequence_replay_buffer.py

作用：
    实现论文第 5 章 LSTM-SAC 使用的固定窗口序列经验回放池。

经验格式：
    state_sequence      当前状态序列 [L, state_dim]
    action              当前动作 [action_dim]
    reward              当前奖励
    next_state_sequence 下一状态序列 [L, state_dim]
    done                当前回合是否结束

说明：
    第一版采用 StateSequenceWrapper 生成固定窗口序列，本 buffer 直接保存序列 transition。
    这样可以稳定完成第 5 章 LSTM-SAC 训练链路，并避免 episode 动态切片带来的额外复杂度。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch


@dataclass
class SequenceReplayBufferConfig:
    """
    SequenceReplayBufferConfig

    作用：
        管理 LSTM-SAC 序列经验池配置。

    属性：
        state_dim：
            单个状态向量维度。
        action_dim：
            动作向量维度。
        capacity：
            经验池最大容量。
        sequence_length：
            每条样本包含的连续状态数量。
        device：
            采样后张量所在设备。
    """

    state_dim: int
    action_dim: int
    capacity: int
    sequence_length: int
    device: str = "cpu"


class SequenceReplayBuffer:
    """
    LSTM-SAC 固定窗口序列经验回放池。

    设计目标：
        1. 与普通 ReplayBuffer 一样使用循环覆盖；
        2. 直接保存包装器输出的 [sequence_length, state_dim] 序列；
        3. sample() 返回 LSTMSACAgent.update() 可直接使用的 batch 字典。
    """

    def __init__(self, config: SequenceReplayBufferConfig) -> None:
        """
        初始化序列经验回放池。

        参数：
            config：
                SequenceReplayBufferConfig 配置对象。
        """
        # state_dim/action_dim：单步状态维度和动作维度。
        self.state_dim = int(config.state_dim)
        self.action_dim = int(config.action_dim)

        # capacity：经验池最大容量。
        self.capacity = int(config.capacity)

        # sequence_length：每条样本的状态序列长度。
        self.sequence_length = int(config.sequence_length)

        # device：采样后张量所在设备。
        self.device = torch.device(config.device)

        if self.state_dim <= 0:
            raise ValueError(f"state_dim 必须大于 0，当前为：{self.state_dim}")

        if self.action_dim <= 0:
            raise ValueError(f"action_dim 必须大于 0，当前为：{self.action_dim}")

        if self.capacity <= 0:
            raise ValueError(f"capacity 必须大于 0，当前为：{self.capacity}")

        if self.sequence_length <= 0:
            raise ValueError(f"sequence_length 必须大于 0，当前为：{self.sequence_length}")

        # state_sequences：当前状态序列，形状为 [capacity, L, state_dim]。
        self.state_sequences = np.zeros(
            (self.capacity, self.sequence_length, self.state_dim),
            dtype=np.float32,
        )

        # actions：当前动作，形状为 [capacity, action_dim]。
        self.actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)

        # rewards：奖励，形状为 [capacity, 1]。
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)

        # next_state_sequences：下一状态序列，形状为 [capacity, L, state_dim]。
        self.next_state_sequences = np.zeros(
            (self.capacity, self.sequence_length, self.state_dim),
            dtype=np.float32,
        )

        # dones：终止标志，形状为 [capacity, 1]。
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)

        # pointer：循环写入位置。
        self.pointer = 0

        # size：当前有效样本数。
        self.size = 0

    def add(
        self,
        state_sequence: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state_sequence: np.ndarray,
        done: bool,
    ) -> None:
        """
        添加一条序列 transition。

        参数：
            state_sequence：
                当前状态序列，形状应为 [sequence_length, state_dim]。
            action：
                当前动作，形状应为 [action_dim]。
            reward：
                当前奖励。
            next_state_sequence：
                下一状态序列，形状应为 [sequence_length, state_dim]。
            done：
                当前回合是否结束。
        """
        # state_sequence_array：转换后的当前状态序列。
        state_sequence_array = np.asarray(state_sequence, dtype=np.float32)

        # next_state_sequence_array：转换后的下一状态序列。
        next_state_sequence_array = np.asarray(next_state_sequence, dtype=np.float32)

        # action_array：转换并拉平后的动作。
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)

        expected_sequence_shape = (self.sequence_length, self.state_dim)

        if state_sequence_array.shape != expected_sequence_shape:
            raise ValueError(
                f"state_sequence 形状错误，应为 {expected_sequence_shape}，"
                f"实际为 {state_sequence_array.shape}"
            )

        if next_state_sequence_array.shape != expected_sequence_shape:
            raise ValueError(
                f"next_state_sequence 形状错误，应为 {expected_sequence_shape}，"
                f"实际为 {next_state_sequence_array.shape}"
            )

        if action_array.shape[0] != self.action_dim:
            raise ValueError(f"action 维度错误，应为 {self.action_dim}，实际为 {action_array.shape[0]}")

        # index：本次写入位置。
        index = self.pointer

        self.state_sequences[index] = state_sequence_array
        self.actions[index] = action_array
        self.rewards[index] = float(reward)
        self.next_state_sequences[index] = next_state_sequence_array
        self.dones[index] = float(done)

        # pointer/size：循环推进经验池状态。
        self.pointer = (self.pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        随机采样一批序列 transition。

        参数：
            batch_size：
                采样数量。

        返回：
            batch：
                包含 state_sequence、action、reward、next_state_sequence、done 的张量字典。
        """
        if self.size == 0:
            raise RuntimeError("序列经验回放池为空，无法采样。")

        if batch_size <= 0:
            raise ValueError(f"batch_size 必须大于 0，当前为：{batch_size}")

        if batch_size > self.size:
            raise ValueError(f"batch_size={batch_size} 大于当前经验池样本数 size={self.size}")

        # indices：随机采样得到的样本索引。
        indices = np.random.randint(low=0, high=self.size, size=batch_size)

        # batch：转换为 torch.Tensor 后的训练 batch。
        batch = {
            "state_sequence": torch.as_tensor(
                self.state_sequences[indices],
                dtype=torch.float32,
                device=self.device,
            ),
            "action": torch.as_tensor(self.actions[indices], dtype=torch.float32, device=self.device),
            "reward": torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=self.device),
            "next_state_sequence": torch.as_tensor(
                self.next_state_sequences[indices],
                dtype=torch.float32,
                device=self.device,
            ),
            "done": torch.as_tensor(self.dones[indices], dtype=torch.float32, device=self.device),
        }

        return batch

    def clear(self) -> None:
        """
        清空序列经验回放池。

        说明：
            只重置 pointer 和 size，不重新分配底层数组。
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
        判断经验池是否已满。
        """
        return self.size >= self.capacity
