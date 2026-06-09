"""
sequence_replay_buffer.py

作用：
    为论文第 5 章 LSTM-SAC 预留序列经验回放池接口。

说明：
    普通 SAC 使用单步经验：
        (s_t, a_t, r_t, s_{t+1}, done_t)

    LSTM-SAC 需要连续状态序列：
        (s_{t-L+1}, ..., s_t)

    后续实现第 5 章时，本文件将用于随机批量序列采样。
"""

from __future__ import annotations

from dataclasses import dataclass


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
            每次采样的连续状态序列长度。

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
    LSTM-SAC 序列经验回放池预留类。

    当前阶段：
        只保留类名和接口位置，不投入训练。

    后续阶段：
        实现连续序列存储、合法序列索引检查、batch 序列采样等功能。
    """

    def __init__(self, config: SequenceReplayBufferConfig) -> None:
        """
        初始化序列经验回放池。

        参数：
            config：
                SequenceReplayBufferConfig 配置对象。
        """
        # config：保存配置对象，便于后续实现时使用。
        self.config = config

        raise NotImplementedError(
            "SequenceReplayBuffer 是为第 5 章 LSTM-SAC 预留的接口，"
            "当前阶段先实现普通 SAC ReplayBuffer。"
        )