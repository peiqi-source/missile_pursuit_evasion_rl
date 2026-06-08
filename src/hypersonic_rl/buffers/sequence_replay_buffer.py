"""第 5 章 LSTM-SAC 序列经验回放缓存预留。"""

from __future__ import annotations

from typing import Any


class SequenceReplayBuffer:
    """状态序列回放缓存骨架。

    后续将按 episode 保存连续 transition，并采样固定长度状态序列。当前阶段普通 SAC
    使用 ReplayBuffer，因此该类只保留接口。
    """

    def __init__(self, sequence_length: int = 8, capacity: int = 100000) -> None:
        self.sequence_length = int(sequence_length)
        self.capacity = int(capacity)
        self.storage: list[Any] = []

    def add_episode(self, episode: Any) -> None:
        """添加一个 episode 序列。"""
        self.storage.append(episode)
        if len(self.storage) > self.capacity:
            self.storage.pop(0)

    def sample(self, batch_size: int) -> Any:
        """采样序列 batch，完整逻辑将在 LSTM-SAC 阶段实现。"""
        raise NotImplementedError("SequenceReplayBuffer 的采样逻辑将在 LSTM-SAC 阶段实现。")

    def __len__(self) -> int:
        """返回已保存 episode 数量。"""
        return len(self.storage)

