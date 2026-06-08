"""普通 SAC 经验回放缓存。"""

from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    """使用 numpy 存储 transition，采样时转换为 torch tensor。"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        capacity: int,
        device: torch.device | str = "cpu",
    ) -> None:
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.ptr = 0
        self.size = 0
        # states：当前状态观测。
        self.states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        # actions：智能体执行的动作。
        self.actions = np.zeros((self.capacity, action_dim), dtype=np.float32)
        # rewards：单步奖励。
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        # next_states：执行动作后的下一状态观测。
        self.next_states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        # dones：episode 是否结束，结束时目标 Q 不再引入下一状态价值。
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """向回放缓存写入一条 transition。"""
        self.states[self.ptr] = np.asarray(state, dtype=np.float32).reshape(-1)
        self.actions[self.ptr] = np.asarray(action, dtype=np.float32).reshape(-1)
        self.rewards[self.ptr] = float(reward)
        self.next_states[self.ptr] = np.asarray(next_state, dtype=np.float32).reshape(-1)
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        """随机采样 batch 并转换到指定设备。"""
        if self.size < batch_size:
            raise ValueError("ReplayBuffer 中样本数量不足，无法采样指定 batch_size。")
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.states[indices], device=self.device),
            torch.as_tensor(self.actions[indices], device=self.device),
            torch.as_tensor(self.rewards[indices], device=self.device),
            torch.as_tensor(self.next_states[indices], device=self.device),
            torch.as_tensor(self.dones[indices], device=self.device),
        )

    def __len__(self) -> int:
        """返回当前缓存中的有效 transition 数量。"""
        return self.size

